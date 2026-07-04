"""RAGコアロジック: 検索器・クエリ変換・リランク・回答生成.

EnterpriseRAG-Bench のサブセット (confluence/jira/github) を対象に,
3系統の検索器 (BM25 / TF-IDF文字n-gram / multilingual-e5-small ベクトル) と,
claude-sonnet-5 を使ったクエリ変換・リランク・回答生成を提供する.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"
MODEL = "claude-sonnet-5"
EMB_MODEL = "intfloat/multilingual-e5-small"
MAX_DOC_CHARS = 4000  # プロンプトに入れる1文書あたりの上限
RRF_K = 60

# 選択可能なデータセット。language はクエリ変換・回答生成プロンプトの切り替えに使う
DATASETS = {
    "EnterpriseRAG-Bench (英語)": {
        "documents": "documents_subset.parquet",
        "questions": "questions_subset.parquet",
        "language": "English",
        "domain_hint": "company internal documents (Confluence, Jira, GitHub)",
        "caption": (
            "コーパス: EnterpriseRAG-Bench サブセット\n\n"
            "Confluence 5,189 / Jira 6,120 / GitHub 8,052 文書 (英語・合成社内文書)"
        ),
        "placeholder": "例: What are the default size limits for file uploads?",
    },
    "Allganize RAG-Evaluation-JA (日本語)": {
        "documents": "documents_allganize_ja.parquet",
        "questions": "questions_allganize_ja.parquet",
        "language": "Japanese",
        "domain_hint": (
            "Japanese business documents (PDF pages) from five domains: "
            "finance, IT, manufacturing, public sector, retail"
        ),
        "caption": (
            "コーパス: Allganize RAG-Evaluation-Dataset-JA\n\n"
            "金融/IT/製造/公共/小売 5ドメインのPDF 65本 (約2,100ページ・日本語)。"
            "1ページ=1文書として検索します。"
        ),
        "placeholder": "例: 火災保険の収益悪化に対し、損害保険各社はどのような対策を講じていますか？",
    },
}


# ---------------------------------------------------------------- utilities

_CJK = (
    "぀-ゟ"   # ひらがな
    "゠-ヿ"   # カタカナ
    "一-鿿"   # CJK統合漢字
    "㐀-䶿"   # CJK拡張A
    "ｦ-ﾟ"   # 半角カナ
)
_TOKEN_RE = re.compile(rf"[a-z0-9]+|[{_CJK}]+")


def _tokenize(text: str) -> list[str]:
    """英数字は単語単位,日本語 (CJK) は文字バイグラムでトークン化する.

    形態素解析器なしで BM25 を日本語に対応させるための簡易実装。
    """
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text.lower()):
        s = m.group(0)
        if s[0].isascii():
            tokens.append(s)
        elif len(s) == 1:
            tokens.append(s)
        else:
            tokens.extend(s[i:i + 2] for i in range(len(s) - 1))
    return tokens


@dataclass
class RetrievalResult:
    doc_id: str
    title: str
    content: str
    source_type: str
    score: float


@dataclass
class RAGTrace:
    """パイプラインの中間状態 (UI表示用).

    各ステップは name / detail に加え,任意で
    - table: 文書リストや統合内訳のテーブル (list[dict], doc_id 列があれば正解判定に使う)
    - llm_in / llm_out: そのステップでのLLMへの入力プロンプトと生の出力
    を持つ。
    """

    steps: list[dict] = field(default_factory=list)

    def add(self, name: str, detail: str = "", *,
            table: list[dict] | None = None,
            llm_in: str | None = None, llm_out: str | None = None) -> None:
        step: dict = {"name": name, "detail": detail}
        if table is not None:
            step["table"] = table
        if llm_in is not None:
            step["llm_in"] = llm_in
        if llm_out is not None:
            step["llm_out"] = llm_out
        self.steps.append(step)


def _preview(text: str, n: int = 120) -> str:
    return re.sub(r"\s+", " ", text)[:n]


def docs_table(results: list[RetrievalResult]) -> list[dict]:
    """検索結果をトレース表示用テーブルに変換する."""
    return [{
        "順位": i, "スコア": round(r.score, 4), "doc_id": r.doc_id,
        "タイトル": r.title, "本文冒頭": _preview(r.content),
    } for i, r in enumerate(results, 1)]


def _token_preview(query: str, n: int = 25) -> str:
    toks = _tokenize(query)
    return " / ".join(toks[:n]) + (" …" if len(toks) > n else "")


# ---------------------------------------------------------------- corpus

_emb_model = None


def _get_emb_model():
    """multilingual-e5-small を遅延ロードする (初回はモデルのダウンロードが走る)."""
    global _emb_model
    if _emb_model is None:
        from sentence_transformers import SentenceTransformer
        _emb_model = SentenceTransformer(EMB_MODEL)
    return _emb_model


class Corpus:
    """文書コーパスと BM25 / TF-IDF / ベクトル (e5) インデックス."""

    def __init__(self, dataset: str = next(iter(DATASETS))):
        spec = DATASETS[dataset]
        self.dataset = dataset
        self.language: str = spec["language"]
        self.domain_hint: str = spec["domain_hint"]
        parquet_path = DATA_DIR / spec["documents"]
        df = pd.read_parquet(parquet_path)
        df["content"] = df["content"].fillna("")
        df["title"] = df["title"].fillna("(no title)")
        self.df = df.reset_index(drop=True)
        self._parquet_path = parquet_path
        self._doc_emb: np.ndarray | None = None

        cache = parquet_path.with_suffix(".index.pkl")
        if cache.exists() and cache.stat().st_mtime > parquet_path.stat().st_mtime:
            with open(cache, "rb") as f:
                self.bm25, self.tfidf, self.tfidf_matrix = pickle.load(f)
            return

        # 検索対象テキストはタイトル+本文
        texts = (df["title"] + "\n" + df["content"]).tolist()
        self.bm25 = BM25Okapi([_tokenize(t) for t in texts])
        # 文字n-gram TF-IDF: 埋め込み不要で表記ゆれにある程度頑健
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), max_features=100_000,
            sublinear_tf=True, dtype=np.float32,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        with open(cache, "wb") as f:
            pickle.dump((self.bm25, self.tfidf, self.tfidf_matrix), f)

    def __len__(self) -> int:
        return len(self.df)

    def _to_results(self, indices, scores) -> list[RetrievalResult]:
        out = []
        for i, s in zip(indices, scores):
            row = self.df.iloc[i]
            out.append(RetrievalResult(
                doc_id=row["doc_id"], title=row["title"],
                content=row["content"], source_type=row["source_type"],
                score=float(s),
            ))
        return out

    def search_bm25(self, query: str, k: int = 10) -> list[RetrievalResult]:
        scores = self.bm25.get_scores(_tokenize(query))
        idx = scores.argsort()[::-1][:k]
        return self._to_results(idx, scores[idx])

    def search_tfidf(self, query: str, k: int = 10) -> list[RetrievalResult]:
        qv = self.tfidf.transform([query])
        sims = cosine_similarity(qv, self.tfidf_matrix)[0]
        idx = sims.argsort()[::-1][:k]
        return self._to_results(idx, sims[idx])

    def _ensure_embeddings(self) -> np.ndarray:
        """文書埋め込みを (無ければ計算して) 返す。.e5.npy にキャッシュする."""
        if self._doc_emb is not None:
            return self._doc_emb
        cache = self._parquet_path.with_suffix(".e5.npy")
        if cache.exists() and cache.stat().st_mtime > self._parquet_path.stat().st_mtime:
            self._doc_emb = np.load(cache)
            return self._doc_emb
        # e5 系は文書側に "passage: " プレフィックスが必要
        texts = ("passage: " + self.df["title"] + "\n" + self.df["content"]).tolist()
        emb = _get_emb_model().encode(
            texts, batch_size=64, normalize_embeddings=True,
            show_progress_bar=True,
        )
        self._doc_emb = np.asarray(emb, dtype=np.float32)
        np.save(cache, self._doc_emb)
        return self._doc_emb

    def search_vector(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """multilingual-e5-small の密ベクトルによる意味検索."""
        doc_emb = self._ensure_embeddings()
        # e5 系はクエリ側に "query: " プレフィックスが必要
        qv = _get_emb_model().encode(
            ["query: " + query], normalize_embeddings=True)[0]
        sims = doc_emb @ qv  # 正規化済みなので内積 = コサイン類似度
        idx = sims.argsort()[::-1][:k]
        return self._to_results(idx, sims[idx])

    def search_hybrid(self, query: str, k: int = 10, pool: int = 50,
                      trace: RAGTrace | None = None) -> list[RetrievalResult]:
        bm = self.search_bm25(query, pool)
        tf = self.search_tfidf(query, pool)
        fused, breakdown = rrf_fuse_detailed([bm, tf], ["BM25", "TF-IDF"], k)
        if trace is not None:
            trace.add(
                "BM25検索",
                f"クエリ: `{query if len(query) < 100 else query[:100] + '…'}`\n\n"
                f"トークン列: `{_token_preview(query)}`\n\n"
                f"候補 {pool} 件を取得 (上位10件を表示)",
                table=docs_table(bm[:10]))
            trace.add(
                "TF-IDF検索 (文字n-gram)",
                f"3〜4文字n-gramのコサイン類似度で候補 {pool} 件を取得 (上位10件を表示)",
                table=docs_table(tf[:10]))
            trace.add(
                "RRF統合",
                f"2系統の候補 各{pool}件を Reciprocal Rank Fusion "
                f"(スコア = Σ 1/({RRF_K}+順位)) で統合し top-{k} を選択。"
                "「-」はその検索器の候補に入らなかったことを示す。",
                table=breakdown)
        return fused


def rrf_fuse(result_lists: list[list[RetrievalResult]], k: int) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion で複数の検索結果を統合する."""
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievalResult] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            scores[r.doc_id] = scores.get(r.doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            by_id.setdefault(r.doc_id, r)
    ranked = sorted(scores, key=lambda d: scores[d], reverse=True)[:k]
    return [RetrievalResult(**{**by_id[d].__dict__, "score": scores[d]}) for d in ranked]


def rrf_fuse_detailed(result_lists: list[list[RetrievalResult]], labels: list[str],
                      k: int) -> tuple[list[RetrievalResult], list[dict]]:
    """RRF統合し,各文書がどのリストの何位から来たかの内訳テーブルも返す."""
    fused = rrf_fuse(result_lists, k)
    ranks = [{r.doc_id: j for j, r in enumerate(lst, 1)} for lst in result_lists]
    breakdown = []
    for i, r in enumerate(fused, 1):
        row: dict = {"順位": i, "RRFスコア": round(r.score, 4), "doc_id": r.doc_id}
        for label, rank_map in zip(labels, ranks):
            # int と "-" の混在は Arrow 変換に失敗するため文字列に統一
            pos = rank_map.get(r.doc_id)
            row[f"{label} 順位"] = str(pos) if pos is not None else "-"
        row["タイトル"] = r.title
        breakdown.append(row)
    return fused, breakdown


# ---------------------------------------------------------------- LLM helpers

def _llm(client: anthropic.Anthropic, prompt: str, max_tokens: int = 1024,
         system: str | None = None) -> str:
    kwargs = {"system": system} if system else {}
    msg = client.messages.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}], **kwargs,
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def _context_block(results: list[RetrievalResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        body = r.content[:MAX_DOC_CHARS]
        parts.append(
            f'<document index="{i}" id="{r.doc_id}" source="{r.source_type}" '
            f'title="{r.title}">\n{body}\n</document>')
    return "\n".join(parts)


def generate_answer(client: anthropic.Anthropic, query: str,
                    results: list[RetrievalResult],
                    domain_hint: str = "company internal documents",
                    trace: "RAGTrace | None" = None) -> str:
    """検索結果を根拠に回答を生成する."""
    system = (
        f"You are an assistant answering questions about {domain_hint}. "
        "Answer based ONLY on the provided documents. Cite document indices "
        "like [1] for every claim. If the documents do not contain the answer, "
        "say so explicitly. "
        "回答は日本語で行うこと(文書からの固有名詞・引用は原文のままでよい)."
    )
    prompt = f"{_context_block(results)}\n\n質問: {query}"
    answer = _llm(client, prompt, max_tokens=1500, system=system)
    if trace is not None:
        trace.add(
            "回答生成",
            f"{MODEL} が {len(results)} 件の文書 (各{MAX_DOC_CHARS}字まで) を"
            f"根拠に回答を生成 (入力 約{len(system) + len(prompt):,} 文字)",
            llm_in=f"<system>\n{system}\n</system>\n\n{prompt}",
            llm_out=answer)
    return answer


# ---------------------------------------------------------------- pipelines

def _parse_json_list(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        return [str(x) for x in json.loads(m.group(0))]
    except json.JSONDecodeError:
        return []


def pipeline_naive_bm25(client, corpus: Corpus, query: str, k: int, trace: RAGTrace):
    results = corpus.search_bm25(query, k)
    trace.add(
        "BM25検索",
        f"トークン列: `{_token_preview(query)}` で top-{k} を取得",
        table=docs_table(results))
    return results


def pipeline_naive_tfidf(client, corpus: Corpus, query: str, k: int, trace: RAGTrace):
    results = corpus.search_tfidf(query, k)
    trace.add(
        "TF-IDF検索",
        f"3〜4文字n-gramのコサイン類似度で top-{k} を取得",
        table=docs_table(results))
    return results


def pipeline_naive_vector(client, corpus: Corpus, query: str, k: int, trace: RAGTrace):
    results = corpus.search_vector(query, k)
    trace.add(
        f"ベクトル検索 ({EMB_MODEL})",
        f"クエリを `query: ` プレフィックス付きで埋め込み,文書埋め込み "
        f"({len(corpus):,}件) とのコサイン類似度で top-{k} を取得",
        table=docs_table(results))
    return results


def pipeline_hybrid(client, corpus: Corpus, query: str, k: int, trace: RAGTrace):
    return corpus.search_hybrid(query, k, trace=trace)


def pipeline_hybrid_vector(client, corpus: Corpus, query: str, k: int,
                           trace: RAGTrace, pool: int = 50):
    bm = corpus.search_bm25(query, pool)
    vec = corpus.search_vector(query, pool)
    trace.add(
        "BM25検索",
        f"トークン列: `{_token_preview(query)}` で候補 {pool} 件を取得 (上位10件を表示)",
        table=docs_table(bm[:10]))
    trace.add(
        f"ベクトル検索 ({EMB_MODEL})",
        f"コサイン類似度で候補 {pool} 件を取得 (上位10件を表示)",
        table=docs_table(vec[:10]))
    fused, breakdown = rrf_fuse_detailed([bm, vec], ["BM25", "ベクトル"], k)
    trace.add(
        "RRF統合",
        f"キーワード (BM25) と意味 (ベクトル) の候補 各{pool}件を "
        f"Reciprocal Rank Fusion で統合し top-{k} を選択。"
        "「-」はその検索器の候補に入らなかったことを示す。",
        table=breakdown)
    return fused


def pipeline_query_rewrite(client, corpus: Corpus, query: str, k: int, trace: RAGTrace):
    prompt = (
        f"Rewrite the following question as a concise {corpus.language} keyword "
        f"search query for searching {corpus.domain_hint}. "
        "Output ONLY the query.\n\n" + query)
    rewritten = _llm(client, prompt).strip()
    trace.add("クエリ書き換え", f"書き換え後クエリ: `{rewritten}`",
              llm_in=prompt, llm_out=rewritten)
    return corpus.search_hybrid(rewritten, k, trace=trace)


def pipeline_hyde(client, corpus: Corpus, query: str, k: int, trace: RAGTrace):
    prompt = (
        f"Write a short hypothetical document (in {corpus.language}, ~150 words) "
        f"in the style of {corpus.domain_hint} that would perfectly answer this "
        "question. Output only the document text.\n\nQuestion: " + query)
    hypo = _llm(client, prompt, max_tokens=400)
    trace.add("HyDE: 仮想文書生成",
              "生成した仮想文書を元クエリに連結し,検索クエリとして使う",
              llm_in=prompt, llm_out=hypo)
    return corpus.search_hybrid(query + "\n" + hypo, k, trace=trace)


def pipeline_rag_fusion(client, corpus: Corpus, query: str, k: int, trace: RAGTrace):
    prompt = (
        f"Generate 4 diverse {corpus.language} search queries for "
        f"{corpus.domain_hint} that together cover this question from different "
        "angles. Output a JSON array of strings only.\n\nQuestion: " + query)
    raw = _llm(client, prompt, max_tokens=300)
    queries = _parse_json_list(raw)[:4] or [query]
    trace.add("RAG-Fusion: 派生クエリ生成",
              "\n".join(f"- {q}" for q in queries), llm_in=prompt, llm_out=raw)
    labeled = [("元クエリ", query)] + [(f"派生{i}", q) for i, q in enumerate(queries, 1)]
    lists = []
    for label, q in labeled:
        res = corpus.search_hybrid(q, 30)
        lists.append(res)
        trace.add(f"Hybrid検索 ({label})",
                  f"`{q}` で候補30件を取得 (上位5件を表示)",
                  table=docs_table(res[:5]))
    results, breakdown = rrf_fuse_detailed(lists, [lb for lb, _ in labeled], k)
    trace.add("RRF統合 (並列検索)",
              f"{len(lists)}本のクエリの検索結果を統合し top-{k}。"
              "「-」はそのクエリの候補30件に入らなかったことを示す。",
              table=breakdown)
    return results


def pipeline_llm_rerank(client, corpus: Corpus, query: str, k: int, trace: RAGTrace):
    pool = corpus.search_hybrid(query, 20, trace=trace)
    listing = "\n".join(
        f"[{i}] ({r.source_type}) {r.title}: {r.content[:300]}"
        for i, r in enumerate(pool))
    prompt = (
        f"Question: {query}\n\nRank the following documents by relevance to the "
        f"question. Output a JSON array of the indices of the top {k} most "
        f"relevant documents, most relevant first. JSON array only.\n\n{listing}")
    raw = _llm(client, prompt, max_tokens=200)
    try:
        order = [int(i) for i in _parse_json_list(raw) if str(i).isdigit()]
        order = [i for i in order if 0 <= i < len(pool)][:k]
    except (ValueError, TypeError):
        order = []
    results = [pool[i] for i in order] if order else pool[:k]
    rerank_table = [{
        "リランク後": i, "第1段順位": j + 1, "doc_id": pool[j].doc_id,
        "タイトル": pool[j].title,
    } for i, j in enumerate(order, 1)] if order else docs_table(results)
    trace.add(
        "LLMリランク",
        "第1段の候補20件 (タイトル+本文300字) をLLMに渡し,関連度順の"
        f"インデックス配列を出力させて上位{k}件を選抜"
        + ("" if order else "。**出力のパースに失敗したため検索順を使用**"),
        llm_in=prompt, llm_out=raw, table=rerank_table)
    return results


def pipeline_crag(client, corpus: Corpus, query: str, k: int, trace: RAGTrace):
    first = corpus.search_hybrid(query, k, trace=trace)
    listing = "\n".join(
        f"[{i}] {r.title}: {r.content[:200]}" for i, r in enumerate(first))
    prompt = (
        f"Question: {query}\n\nRetrieved documents:\n{listing}\n\n"
        'Do these documents contain enough information to answer the question? '
        'Reply with JSON: {"sufficient": true/false, "revised_query": "..."} '
        f"where revised_query is a better {corpus.language} search query "
        "if insufficient.")
    verdict = _llm(client, prompt, max_tokens=200)
    m = re.search(r"\{.*\}", verdict, re.DOTALL)
    sufficient, revised = True, None
    if m:
        try:
            j = json.loads(m.group(0))
            sufficient = bool(j.get("sufficient", True))
            revised = j.get("revised_query")
        except json.JSONDecodeError:
            pass
    trace.add(
        "CRAG: 検索結果の評価",
        f"判定: sufficient={sufficient}"
        + (f", 改訂クエリ: `{revised}`" if revised and not sufficient else "")
        + ("。十分と判定されたため初回の検索結果をそのまま使う" if sufficient else ""),
        llm_in=prompt, llm_out=verdict)
    if not sufficient and revised:
        second = corpus.search_hybrid(str(revised), k)
        trace.add("再検索", f"改訂クエリ `{revised}` で top-{k} を取得",
                  table=docs_table(second))
        results, breakdown = rrf_fuse_detailed([first, second], ["初回", "再検索"], k)
        trace.add("RRF統合 (初回+再検索)",
                  f"初回と再検索の結果を統合し top-{k}", table=breakdown)
        return results
    return first


PIPELINES = {
    "Naive RAG (BM25)": pipeline_naive_bm25,
    "Naive RAG (TF-IDF)": pipeline_naive_tfidf,
    "Naive RAG (ベクトル検索)": pipeline_naive_vector,
    "Hybrid + RRF": pipeline_hybrid,
    "Hybrid (BM25+ベクトル) + RRF": pipeline_hybrid_vector,
    "Query Rewriting": pipeline_query_rewrite,
    "HyDE": pipeline_hyde,
    "RAG-Fusion": pipeline_rag_fusion,
    "LLM Rerank": pipeline_llm_rerank,
    "Corrective RAG (簡易版)": pipeline_crag,
}

PIPELINE_DESCRIPTIONS = {
    "Naive RAG (BM25)": "キーワードマッチ (BM25) で検索してそのまま生成。ベースライン。",
    "Naive RAG (TF-IDF)": "文字n-gram TF-IDF のコサイン類似度で検索。表記ゆれにやや頑健。",
    "Naive RAG (ベクトル検索)": (
        f"密ベクトル ({EMB_MODEL}) のコサイン類似度で検索。"
        "キーワードが一致しなくても意味的に近い文書を拾える。"
    ),
    "Hybrid + RRF": "BM25 と TF-IDF の結果を Reciprocal Rank Fusion で統合。",
    "Hybrid (BM25+ベクトル) + RRF": (
        "キーワード (BM25) と意味 (ベクトル) の結果を RRF で統合。"
        "実務で定番のハイブリッド構成。"
    ),
    "Query Rewriting": "LLMがクエリを検索向けの英語キーワードに書き換えてから Hybrid 検索。",
    "HyDE": "LLMが仮想的な回答文書を生成し,それをクエリに加えて検索。",
    "RAG-Fusion": "LLMが4本の派生クエリを生成し,並列検索の結果を RRF で統合。",
    "LLM Rerank": "Hybrid で候補20件を取り,LLMが関連度順に並び替えて上位k件を選抜。",
    "Corrective RAG (簡易版)": "検索結果の十分性をLLMが評価し,不十分ならクエリを改訂して再検索。",
}


def run_pipeline(method: str, client: anthropic.Anthropic, corpus: Corpus,
                 query: str, k: int = 5) -> tuple[str, list[RetrievalResult], RAGTrace]:
    """検索パイプラインを実行し (回答, 検索結果, トレース) を返す."""
    trace = RAGTrace()
    results = PIPELINES[method](client, corpus, query, k, trace)
    answer = generate_answer(client, query, results, corpus.domain_hint, trace)
    return answer, results, trace


def load_questions(dataset: str = next(iter(DATASETS))) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / DATASETS[dataset]["questions"])


if __name__ == "__main__":
    # 全データセットの検索インデックスと文書埋め込みを事前構築する
    # (アプリ初回起動・初回ベクトル検索を速くするため):
    #   python rag_core.py
    for name in DATASETS:
        print(f"=== {name} ===")
        corpus = Corpus(name)
        print(f"文書数: {len(corpus):,} / 埋め込みを構築中 ({EMB_MODEL}) ...")
        emb = corpus._ensure_embeddings()
        print(f"完了: shape={emb.shape}")
