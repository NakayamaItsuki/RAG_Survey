# RAG_Survey

- RAG手法を体系的に整理し,特徴や長所・短所を Survey.md にまとめる 
- 主要なRAG手法を手元で動かし,動作を確認する
- EnterpriseRAG-Benchで社内ドキュメントの検索を再現
- 日本語データセット Allganize RAG-Evaluation-Dataset-JA にも対応(デモアプリで切り替え可能)
- Streamlitでデモアプリを作成.RAG手法を選択し,クエリを入力すると実際に検索&回答生成を行う.
- LLMは claude-sonnet-5 を使用する.APIキーは .streamlit/secrets.toml に記載.

---

## 構成

| ファイル | 内容 |
|---|---|
| `Survey.md` | RAG手法のSurvey(パラダイム,検索器,クエリ変換,自己制御型,GraphRAG,評価,比較表) |
| `app.py` | Streamlit デモアプリ |
| `rag_core.py` | RAGコアロジック(検索器・10種のパイプライン・回答生成) |
| `data/documents_subset.parquet` | EnterpriseRAG-Bench サブセット(Confluence/Jira/GitHub, 19,361文書) |
| `data/questions_subset.parquet` | 上記3ソースで完結するベンチマーク質問178問(正解文書ID・正解回答付き) |
| `data/questions.parquet` | ベンチマーク全質問500問 |
| `build_allganize_ja.py` | Allganize RAG-Evaluation-JA の PDF取得・parquet変換スクリプト |
| `data/documents_allganize_ja.parquet` | Allganize JA コーパス(PDF 65本を1ページ=1文書化,2,126文書) |
| `data/questions_allganize_ja.parquet` | Allganize JA ベンチマーク質問300問(正解ページID・正解回答付き) |

## セットアップと実行

必要なもの: Python 3.12 / [uv](https://docs.astral.sh/uv/) / [Anthropic APIキー](https://console.anthropic.com/)(回答生成に使用,従量課金)

```bash
git clone https://github.com/NakayamaItsuki/RAG_Survey.git
cd RAG_Survey

# 1. 仮想環境と依存パッケージ
uv venv .venv --python 3.12
# GPUなし環境は先にCPU版torchを入れるとダウンロードが軽い (CUDA版は約2GB)
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt

# 2. APIキーの設定 (このファイルはgit管理外)
mkdir -p .streamlit
echo 'ANTHROPIC_API_KEY = "sk-ant-..."' > .streamlit/secrets.toml

# 3. 起動
.venv/bin/streamlit run app.py
```

- 検索用のコーパス (parquet) と文書埋め込み (`data/*.e5.npy`) はリポジトリに
  含まれるため,データセットの再構築や埋め込みの事前計算は不要
- 初回起動時のみ BM25 / TF-IDF インデックスを構築する(約90秒。`data/*.index.pkl` に
  キャッシュされ,2回目以降は数秒で起動)。ベクトル検索の初回は HuggingFace から
  埋め込みモデル (約120MB) のダウンロードが走る
- parquet を作り直した場合は対応する `data/*.e5.npy` を削除すること
  (`python rag_core.py` で再構築できる)

## デモアプリで選べるRAG手法

1. **Naive RAG (BM25)** — キーワード検索ベースライン
2. **Naive RAG (TF-IDF)** — 文字n-gramコサイン類似度
3. **Naive RAG (ベクトル検索)** — multilingual-e5-small の密ベクトルによる意味検索
4. **Hybrid + RRF** — BM25 + TF-IDF を Reciprocal Rank Fusion で統合
5. **Hybrid (BM25+ベクトル) + RRF** — キーワードと意味のハイブリッド(実務で定番の構成)
6. **Query Rewriting** — LLMがクエリを検索向けに書き換え
7. **HyDE** — LLMが仮想回答文書を生成して検索
8. **RAG-Fusion** — 派生クエリ4本の並列検索 + RRF
9. **LLM Rerank** — 候補20件をLLMが関連度順に再ランク
10. **Corrective RAG (簡易版)** — 検索結果の十分性をLLMが評価し,不十分なら再検索

ベンチマーク質問を選ぶと,正解文書の再現率 (Recall) と gold answer が表示され,手法間の検索品質を比較できる.
質問タイプ(basic / semantic / constrained など)によって手法の得手不得手が観察できる
(例: 語彙の言い換えを含む semantic 問題は BM25 単体では正解文書を取りこぼしやすい).

### 検索器の性能 (Recall@5)

| 検索器 | EN (178問) | JA (300問) |
|---|---|---|
| BM25 | 0.604 | 0.857 |
| TF-IDF (文字n-gram) | 0.562 | - |
| ベクトル (multilingual-e5-small) | 0.502 | 0.793 |
| Hybrid (BM25+TF-IDF, RRF) | 0.604 | - |
| Hybrid (BM25+ベクトル, RRF) | 0.608 | 0.857 |

語彙ベース検索器同士のハイブリッド (BM25+TF-IDF) は誤りに相関があるため
BM25 単体を超えられていない。ベクトル検索は単体では BM25 に及ばない
(EN側は平均6,300字の文書を512トークンで切り詰めることも不利に働く)が,
誤り方がキーワード検索と異なるため,BM25 とのハイブリッドは両データセットで
単体最良と同等以上になる。LLMによるクエリ変換・リランクを加えたときの改善は,
デモアプリ上で質問ごとに確認できる.

## 実験設定 (RAG手法以外の共通設定)

手法間の比較条件を揃えるため,以下はすべての手法で共通。値はコード中の定数
(`rag_core.py` / `build_allganize_ja.py`) と対応する。

### コーパス構築とチャンク分割

| 項目 | EnterpriseRAG-Bench | Allganize RAG-Evaluation-JA |
|---|---|---|
| 元データ | HuggingFace 配布の parquet | 配布元URLの公開PDF 65本(リンク切れは Wayback Machine から補完) |
| PDF読み取り | -(テキストで配布) | `pypdf`(AES暗号化PDFは `cryptography` で復号) |
| OCR | - | RapidOCR 日本語モデル(ONNX, CPU)。`pypdfium2` で2倍スケール(≒144dpi)にレンダリングして入力 |
| OCRの適用条件 | - | 文書全体のCJK文字比率 < 5% → 全ページOCR(CIDマッピング破損・画像PDF対策)。それ以外でも本文20字未満のページは個別にOCR |
| チャンク分割 | **なし**(ベンチマーク配布の1文書=1検索単位, 平均約6,300字) | **1ページ=1チャンク**(`doc_id = "ファイル名#pページ番号"`, 平均約800字)。ページ内の再分割はしない |
| 正規化 | - | ファイル名を NFC 正規化(documents.csv 側が NFD 混在のため) |
| 検索対象テキスト | タイトル + 改行 + 本文(両データセット共通) | 同左 |

### トークナイズと検索器

| 項目 | 設定 |
|---|---|
| トークナイザ | 小文字化した英数字連続 `[a-z0-9]+` を1トークン。CJK文字(ひらがな/カタカナ/漢字/半角カナ)の連続は**文字バイグラム**に分割。形態素解析器は不使用 |
| BM25 | `rank_bm25.BM25Okapi` のデフォルト(k1=1.5, b=0.75, ε=0.25) |
| TF-IDF | `char_wb` の3〜4文字n-gram, `max_features=100,000`, `sublinear_tf=True`, float32。類似度はコサイン |
| ベクトル検索 | `intfloat/multilingual-e5-small`(384次元)。クエリに `query: `,文書に `passage: ` プレフィックスを付与し,正規化埋め込みの内積(=コサイン類似度)。入力は512トークンで切り詰め |
| インデックスキャッシュ | `data/*.index.pkl`(BM25/TF-IDF, parquetより新しければ再利用)と `data/*.e5.npy`(文書埋め込み, リポジトリ同梱。文書数が一致すれば再利用) |

### 検索・統合のハイパーパラメータ

| 項目 | 値 |
|---|---|
| 取得文書数 top-k | アプリで 3〜10(デフォルト 5) |
| RRF 定数 | k=60(スコア = Σ 1/(60+順位)) |
| Hybrid の候補プール | 各検索器 50件 |
| RAG-Fusion | 派生クエリ4本 + 元クエリ,各クエリ 30件を RRF 統合 |
| LLM Rerank の第1段候補 | Hybrid 20件(LLMにはタイトル+本文300字を提示) |
| CRAG の評価入力 | 各文書のタイトル+本文200字 |

### 生成 (LLM)

| 項目 | 値 |
|---|---|
| モデル | `claude-sonnet-5`(回答生成・クエリ変換・リランク・評価すべて同一) |
| temperature | 未指定(APIデフォルト) |
| 回答生成 | max_tokens=1500。コンテキストは top-k 文書を各**4,000字**まで,`<document index/id/source/title>` タグで区切って投入。文書のみを根拠に [n] 形式で引用させ,回答は日本語を指示 |
| 補助タスクの max_tokens | クエリ書き換え 1024 / HyDE 400 / RAG-Fusion 300 / リランク・CRAG評価 200 |
| クエリ変換の出力言語 | データセットの言語に追従(EN: 英語, JA: 日本語) |

## Allganize RAG-Evaluation-Dataset-JA について

Allganize による日本語RAGベンチマーク
([HuggingFace](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-JA))。
金融/IT/製造/公共/小売の5ドメインの公開PDF 65本(約2,100ページ)と300問のQAからなり,
各質問に正解ページと正解回答が付く。本リポジトリでは1ページ=1文書
(`doc_id = "ファイル名#pページ番号"`)としてparquet化し,デモアプリのデータセット
セレクタから選択できる。日本語検索のため BM25 のトークナイザはCJK文字バイグラムを使う
(形態素解析器は不要)。

parquet を再生成する場合(PDFのダウンロードを含む,数分):

```bash
.venv/bin/python build_allganize_ja.py
```

一部の配布元URLはリンク切れのため,スクリプトは失敗したPDFをスキップする
(その場合,該当質問は除外される)。リンク切れ分は Wayback Machine から
`data/allganize_ja_pdfs/` に手動配置すれば取り込まれる。
テキスト層が無い/フォントのCIDマッピングが壊れているPDF (4本,約100ページ) は
RapidOCR (日本語モデル) で自動的にOCR抽出される。

## EnterpriseRAG-Bench について

Onyx による企業内部ドキュメント特化のRAGベンチマーク
([arXiv:2605.05253](https://arxiv.org/abs/2605.05253) /
[GitHub](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) /
[HuggingFace](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench))。
全体は9ソース約50万文書・500問。本リポジトリでは Confluence + Jira + GitHub の3ソース
(19,361文書)と,この3ソースで完結する178問をサブセットとして使用している。
詳細は `Survey.md` §12 を参照。
