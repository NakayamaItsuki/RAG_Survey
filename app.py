"""RAG手法デモアプリ (Streamlit).

データセット (EnterpriseRAG-Bench サブセット / Allganize RAG-Evaluation-JA) と
RAG手法を選択してクエリを実行し,検索結果と claude-sonnet-5 の回答を表示する.
"""

import time
import tomllib
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

from rag_core import (
    DATASETS,
    PIPELINE_DESCRIPTIONS,
    PIPELINES,
    Corpus,
    load_questions,
    run_pipeline,
)

st.set_page_config(page_title="RAG Survey Demo", page_icon="🔎", layout="wide")


def load_api_key() -> str:
    return st.secrets["ANTHROPIC_API_KEY"]


@st.cache_resource(show_spinner="コーパスを読み込み中 (初回のみ ~1分)...")
def get_corpus(dataset: str) -> Corpus:
    return Corpus(dataset)


@st.cache_data
def get_questions(dataset: str):
    return load_questions(dataset)


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=load_api_key())


# ---------------------------------------------------------------- sidebar

st.sidebar.title("🔎 RAG Survey Demo")

dataset = st.sidebar.selectbox("データセット", list(DATASETS.keys()))
st.sidebar.caption(DATASETS[dataset]["caption"])

method = st.sidebar.radio("RAG手法", list(PIPELINES.keys()))
st.sidebar.info(PIPELINE_DESCRIPTIONS[method])
top_k = st.sidebar.slider("取得文書数 (top-k)", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("各手法の解説はリポジトリ内の `Survey.md` を参照")

# ---------------------------------------------------------------- main

st.title("RAG手法 比較デモ")
st.markdown(
    f"**選択中の手法: {method}** — 質問を入力すると,検索した社内文書を根拠に "
    "`claude-sonnet-5` が回答します。ベンチマーク質問には正解文書IDが付いており,"
    "検索のヒット/ミスが確認できます。"
)

questions = get_questions(dataset)

with st.expander(f"📋 ベンチマーク質問から選ぶ ({len(questions)}問)"):
    qtype = st.selectbox(
        "質問タイプ", ["(すべて)"] + sorted(questions["question_type"].unique())
    )
    filtered = (
        questions
        if qtype == "(すべて)"
        else questions[questions["question_type"] == qtype]
    )
    sel = st.selectbox(
        "質問", filtered["question"].tolist(), index=None, placeholder="質問を選択..."
    )
    if sel is not None:
        st.session_state["query_input"] = sel

query = st.text_area(
    "クエリ", key="query_input", height=80, placeholder=DATASETS[dataset]["placeholder"]
)

run = st.button("🚀 検索 & 回答生成", type="primary", disabled=not query)

if run and query:
    corpus = get_corpus(dataset)
    client = get_client()

    t0 = time.time()
    with st.spinner(f"{method} を実行中..."):
        try:
            answer, results, trace = run_pipeline(method, client, corpus, query, top_k)
        except anthropic.APIError as e:
            st.error(f"Anthropic API エラー: {e}")
            st.stop()
    elapsed = time.time() - t0

    # ベンチマーク質問なら正解情報を取得
    gold_row = questions[questions["question"] == query]
    gold_ids = set(gold_row.iloc[0]["expected_doc_ids"]) if len(gold_row) else set()

    col_ans, col_ret = st.columns([1, 1])

    with col_ans:
        st.subheader("💬 回答")
        st.markdown(answer)
        st.caption(f"⏱ {elapsed:.1f} 秒 / 手法: {method} / top-{top_k}")
        if len(gold_row):
            with st.expander("🏅 ベンチマークの正解 (gold answer)"):
                st.markdown(gold_row.iloc[0]["gold_answer"])

    with col_ret:
        st.subheader("📚 検索結果")
        if gold_ids:
            hit = sum(1 for r in results if r.doc_id in gold_ids)
            st.metric("正解文書の再現率 (Recall)", f"{hit}/{len(gold_ids)}")
        for i, r in enumerate(results, 1):
            mark = ""
            if gold_ids:
                mark = " ✅正解文書" if r.doc_id in gold_ids else ""
            with st.expander(f"[{i}] ({r.source_type}) {r.title}{mark}"):
                st.caption(f"doc_id: `{r.doc_id}` / score: {r.score:.4f}")
                st.text(r.content[:2000])

    st.subheader("🔬 パイプラインのトレース")

    def _clip(s: str, n: int = 8000) -> str:
        return s if len(s) <= n else s[:n] + f"\n... (省略: 全{len(s):,}文字)"

    for i, step in enumerate(trace.steps, 1):
        with st.expander(f"Step {i}: {step['name']}", expanded=False):
            if step.get("detail"):
                st.markdown(step["detail"])
            if step.get("table"):
                df = pd.DataFrame(step["table"])
                if gold_ids and "doc_id" in df.columns:
                    df.insert(
                        0,
                        "正解",
                        df["doc_id"].isin(gold_ids).map({True: "✅", False: ""}),
                    )
                st.dataframe(df, hide_index=True, width="stretch")
            if step.get("llm_in") is not None:
                st.markdown("**📥 LLMへの入力**")
                st.code(_clip(step["llm_in"]), language=None, wrap_lines=True)
            if step.get("llm_out") is not None:
                st.markdown("**📤 LLMの出力 (生テキスト)**")
                st.code(_clip(step["llm_out"]), language=None, wrap_lines=True)
