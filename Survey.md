# RAG手法 Survey

Retrieval-Augmented Generation (RAG) の主要手法を体系的に整理し,それぞれの特徴・長所・短所をまとめる.
最後に本リポジトリのデモアプリ (`app.py`) で実装した手法との対応を示す.

---

## 目次

1. [RAGとは](#1-ragとは)
2. [RAGの基本構成](#2-ragの基本構成)
3. [パラダイムの変遷: Naive → Advanced → Modular](#3-パラダイムの変遷)
4. [検索器 (Retriever) の手法](#4-検索器-retriever-の手法)
5. [クエリ変換系の手法](#5-クエリ変換系の手法)
6. [インデックス・チャンク戦略](#6-インデックスチャンク戦略)
7. [自己制御・適応型RAG](#7-自己制御適応型rag)
8. [構造化知識を使うRAG (GraphRAG)](#8-構造化知識を使うrag)
9. [Agentic RAG](#9-agentic-rag)
10. [Long-Context LLM と RAG](#10-long-context-llm-と-rag)
11. [RAGの評価](#11-ragの評価)
12. [EnterpriseRAG-Bench](#12-enterpriserag-bench)
13. [手法比較表](#13-手法比較表)
14. [デモアプリとの対応](#14-デモアプリとの対応)
15. [参考文献](#15-参考文献)

---

## 1. RAGとは

RAG (Retrieval-Augmented Generation) は,LLMの生成時に**外部知識ベースから関連文書を検索して文脈に注入する**枠組みである (Lewis et al., 2020)。LLM単体の弱点を補う:

- **ハルシネーション低減**: 根拠文書に基づいた回答を生成できる
- **知識の鮮度**: モデルの再学習なしに知識ベースの更新だけで最新情報に追従できる
- **プライベート知識**: 学習データに含まれない社内文書・専門文書を扱える
- **出典提示**: 回答の根拠となった文書を提示でき,検証可能性が上がる

トレードオフとして,検索品質が回答品質の上限を決める(「検索で拾えなかった情報は答えられない」),パイプラインが複雑化する,レイテンシとコストが増える,という課題がある。

## 2. RAGの基本構成

典型的なRAGは3段階からなる:

```
[Indexing]   文書 → チャンク分割 → ベクトル化/索引化 → インデックス格納
[Retrieval]  クエリ → (変換) → 検索 → (リランク) → 上位k件のチャンク
[Generation] クエリ + 検索結果 → プロンプト構築 → LLM → 回答
```

各段階に多数の改善手法が提案されており,本Surveyでは「どの段階を改善するか」の軸で整理する。

## 3. パラダイムの変遷

Gao et al. (2023) のサーベイに従い,RAGは3つのパラダイムに大別される。

### 3.1 Naive RAG

最も素朴な構成。文書を固定長チャンクに分割し,埋め込みベクトル(または疎ベクトル)の類似度で top-k を取得し,そのままプロンプトに連結して生成する。

- **長所**: 実装が単純・低コスト・低レイテンシ。ベースラインとして必須
- **短所**: クエリと文書の語彙・抽象度のミスマッチに弱い。無関係チャンクの混入(precision低下),必要チャンクの取り逃し(recall低下)がそのまま回答品質に響く。マルチホップ質問に弱い

### 3.2 Advanced RAG

Naive RAG の前後に処理を追加して検索品質を上げる。

- **Pre-Retrieval(検索前)**: クエリ書き換え,クエリ拡張,HyDE,メタデータフィルタ
- **Post-Retrieval(検索後)**: リランキング,文脈圧縮,冗長チャンクの除去

- **長所**: 検索のprecision/recallが大きく改善。既存パイプラインへの追加が容易
- **短所**: LLM呼び出し回数が増えレイテンシ・コスト増。パラメータ(k, チャンクサイズ等)の調整項目が増える

### 3.3 Modular RAG

検索・生成・記憶・ルーティングなどをモジュールとして組み替え可能にした一般化。検索と生成を反復するループ(Iterative / Recursive Retrieval)や,クエリの種類に応じて検索経路を切り替えるルーティングを含む。

- **長所**: タスクに応じた柔軟な構成が可能。Self-RAG や Agentic RAG を包含する枠組み
- **短所**: 設計・運用・評価の複雑さが最も高い

## 4. 検索器 (Retriever) の手法

### 4.1 疎ベクトル検索 (Sparse Retrieval)

**BM25 / TF-IDF**。単語の出現頻度に基づくキーワードマッチ。

- **長所**: 高速・安価・解釈性が高い。固有名詞・型番・エラーコードなど**表層一致が重要なクエリに強い**。インデックス構築に学習不要
- **短所**: 同義語・言い換えに弱い(語彙ミスマッチ問題)。意味的な類似を捉えられない

### 4.2 密ベクトル検索 (Dense Retrieval)

DPR (Karpukhin et al., 2020) 以降主流の手法。バイエンコーダでクエリと文書を同一ベクトル空間に埋め込み,近似最近傍探索 (ANN) で検索する。OpenAI text-embedding-3,Cohere embed,BGE,E5 などの埋め込みモデルを利用。

- **長所**: 同義語・言い換え・多言語など**意味的な類似**を捉えられる
- **短所**: 埋め込みモデルのドメイン適合性に依存。固有名詞・数値・ドメイン固有語に弱いことがある。インデックス構築コスト(埋め込み計算)が必要

### 4.3 ハイブリッド検索 + RRF

疎検索と密検索を併用し,Reciprocal Rank Fusion (RRF) 等で順位を統合する。

```
RRF(d) = Σ_r 1 / (k + rank_r(d))   (k ≈ 60)
```

- **長所**: 疎・密の弱点を相互補完し,実務では単独手法より安定して高精度。RRFはスコアの正規化が不要で頑健
- **短所**: 2系統のインデックス維持が必要。融合の重み調整というチューニング項目が増える

### 4.4 Late Interaction (ColBERT)

トークン単位の埋め込みを保持し,MaxSim でクエリ・文書トークン間の細粒度マッチングを行う。

- **長所**: バイエンコーダより高精度で,クロスエンコーダより高速
- **短所**: インデックスサイズが大きい(トークン毎にベクトルを保持)。運用インフラが限られる

### 4.5 リランキング (Reranking)

第1段検索で取得した候補(例: 上位50件)を,より高精度なモデルで並び替えて上位k件に絞る。

- **クロスエンコーダ**: クエリと文書を連結して関連度を直接推定(Cohere Rerank, BGE-reranker等)
- **LLMリランク**: LLMに候補の関連度を採点/選抜させる(RankGPT等)

- **長所**: 検索precisionが大幅に向上。第1段は recall 重視・第2段は precision 重視という役割分担ができる
- **短所**: 候補件数に比例したレイテンシ・コスト増。LLMリランクは特に高コスト

## 5. クエリ変換系の手法

ユーザの生クエリは検索に最適とは限らない。検索前にクエリを変換する。

### 5.1 Query Rewriting(クエリ書き換え)

LLMで口語的・曖昧なクエリを検索向きの表現に書き換える。会話履歴を踏まえた照応解決(「それってどうやるの?」→ 具体的なクエリ)も含む。

- **長所**: 実装が簡単で効果が安定。対話型RAGではほぼ必須
- **短所**: 書き換えで意図がずれるリスク。LLM呼び出し1回分のコスト

### 5.2 HyDE (Hypothetical Document Embeddings)

Gao et al. (2022)。クエリから**仮想的な回答文書**をLLMに生成させ,その文書をクエリ代わりに検索する。「質問と文書」より「文書と文書」の方が類似空間上で近いという発想。

- **長所**: クエリと文書の抽象度ギャップを埋める。ゼロショットで dense retrieval の精度を改善
- **短所**: LLMが誤った仮説を生成すると検索が誤誘導される(ドメイン知識がない領域で顕著)。レイテンシ増

### 5.3 RAG-Fusion / Multi-Query

元クエリからLLMで**複数の視点の派生クエリ**を生成し,並列に検索して RRF で統合する。

- **長所**: 曖昧・多面的な質問で recall が向上。1クエリの言い回し依存が減る
- **短所**: 検索回数が倍増しレイテンシ増。派生クエリの品質に依存

### 5.4 Step-Back Prompting / クエリ分解

具体的な質問を一段抽象化した質問に変換して背景知識を検索する (Step-Back),あるいは複合質問をサブ質問に分解してそれぞれ検索する (Decomposition)。

- **長所**: マルチホップ・複合質問への対応力が上がる
- **短所**: パイプラインが複雑化。サブ質問の統合ロジックが必要

## 6. インデックス・チャンク戦略

### 6.1 チャンク分割

固定長分割,文/段落境界での分割,オーバーラップ付き分割,Markdown/HTML構造を使った分割など。チャンクサイズは「検索の当てやすさ(小)」と「生成に必要な文脈量(大)」のトレードオフ。

### 6.2 Small-to-Big (Parent Document / Sentence Window)

**小さい単位で検索し,大きい単位を生成に渡す**。文単位で索引を張り,ヒットしたら親チャンク(または前後の窓)を返す。

- **長所**: 検索精度と生成文脈量を両立できる。実務で効果が高い定番
- **短所**: インデックス構造が複雑化(親子関係の管理)

### 6.3 RAPTOR(階層的要約インデックス)

Sarthi et al. (2024)。チャンクをクラスタリング→要約を再帰的に繰り返し,要約の木を構築。抽象的な質問には上位ノード,具体的な質問には葉ノードがヒットする。

- **長所**: 「文書全体の要旨」のような抽象度の高い質問に対応できる
- **短所**: インデックス構築コストが高い(大量のLLM要約)。更新に弱い

### 6.4 メタデータ・コンテキスト付与

チャンクにタイトル・日付・ソース種別などのメタデータを付与してフィルタや検索に利用する。Anthropic の **Contextual Retrieval** (2024) は,各チャンクに「文書全体の中でのこのチャンクの位置づけ」をLLMで生成して前置してから索引化する手法で,検索失敗率を大きく削減したと報告している。

- **長所**: チャンク単体では失われる文脈(代名詞,暗黙の主語)を補い,検索精度が向上
- **短所**: インデックス構築時にチャンク数分のLLM呼び出しが必要(プロンプトキャッシュで緩和可能)

## 7. 自己制御・適応型RAG

「いつ検索するか」「検索結果は信頼できるか」をシステム自身に判断させる系統。

### 7.1 Self-RAG

Asai et al. (2023)。リフレクショントークンを出力するよう学習したモデルが,**検索の要否判断 → 検索結果の関連性評価 → 生成 → 自己批評**を行う。

- **長所**: 不要な検索を省き,関連しない検索結果を棄却できる。回答の根拠付けが強化される
- **短所**: 専用の学習が必要(既製LLMへの適用はプロンプトによる模倣になる)。推論が複雑

### 7.2 CRAG (Corrective RAG)

Yan et al. (2024)。軽量な評価器が検索結果を {Correct / Incorrect / Ambiguous} に分類し,不良なら**Web検索などの代替ソースへフォールバック**したり,検索結果から必要部分のみ抽出(decompose-then-recompose)して使う。

- **長所**: 検索失敗時のロバスト性が上がる。既存RAGにプラグインしやすい
- **短所**: 評価器の精度に依存。フォールバック分のレイテンシ増

### 7.3 Adaptive-RAG

Jeong et al. (2024)。クエリの複雑さを分類器で判定し,「検索なし / 1回検索 / 反復検索」を切り替える。

- **長所**: 簡単な質問に重いパイプラインを使わず,コストと精度のバランスを最適化
- **短所**: 複雑さ分類器の学習・調整が必要

### 7.4 FLARE (Forward-Looking Active Retrieval)

Jiang et al. (2023)。生成を進めながら,**確信度の低い文が出そうになったらその都度検索**する能動的検索。

- **長所**: 長文生成で必要な時に必要な知識を取りに行ける
- **短所**: 生成と検索の交互実行でレイテンシが大きい。確信度推定にlogitsアクセスが必要

## 8. 構造化知識を使うRAG

### 8.1 GraphRAG

Microsoft (Edge et al., 2024)。文書からLLMで**エンティティと関係を抽出して知識グラフを構築**し,コミュニティ検出+コミュニティ単位の要約を事前生成する。質問時はグローバル(コミュニティ要約を集約)またはローカル(エンティティ近傍)に検索する。

- **長所**: 「この文書群の主要テーマは?」のような**全体俯瞰型 (global) の質問**に強い。エンティティ間の関係を跨ぐマルチホップ質問に強い
- **短所**: インデックス構築のLLMコストが非常に高い。グラフの更新・保守が重い。単純なfact検索ではNaive RAGと差が出にくい

### 8.2 KG-RAG(既存知識グラフの活用)

既存のナレッジグラフ(社内マスタ,オントロジー)をクエリ(Text-to-Cypher/SPARQL)して構造化知識を取得し,文脈に注入する。

- **長所**: 正確な構造化データ(数値・関係)を直接取得でき,ハルシネーションが起きにくい
- **短所**: 高品質なKGの存在が前提。クエリ生成の失敗モードがある

## 9. Agentic RAG

検索を**ツールとしてエージェントに渡し,計画・実行・観察のループ**で複数回の検索・推論を行わせる (ReAct スタイル)。マルチホップ質問,複数ソースの突き合わせ,「情報が存在しない」ことの判断などに対応する。2024年以降,function calling の成熟とともに実務の主流になりつつある。

- **長所**: 検索回数・戦略をLLMが動的に決定。複雑な質問への対応力が最も高い。「見つからなければ検索語を変えて再検索」が自然にできる
- **短所**: レイテンシ・コストが最も高い。挙動が非決定的で評価・デバッグが難しい。ループの暴走対策(上限設定)が必要

## 10. Long-Context LLM と RAG

コンテキスト長が100万トークン級に伸び,「全部突っ込めばRAG不要では?」という議論がある。現状の整理:

- **Long-Context の強み**: 検索の失敗モードがない。文書横断の総合的理解
- **RAGの強み**: コスト(入力トークン課金は文書量に比例),レイテンシ,数百万文書級へのスケール,出典の明確さ,アクセス制御
- **実務の答え**: 併用。RAGで候補を絞り,余裕のあるコンテキストに多め(数十チャンク)に詰める。"Lost in the middle"(長文中央の情報を見落とす現象)は新しいモデルでは緩和傾向だが,依然としてリランクで上位に重要情報を置く価値はある

## 11. RAGの評価

### 評価の3軸(RAG Triad)

| 軸 | 問い | 代表指標 |
|---|---|---|
| Context Relevance | 検索結果はクエリに関連するか | Precision@k, Recall@k, MRR, nDCG |
| Groundedness / Faithfulness | 回答は検索結果に忠実か | RAGAS Faithfulness, 引用検証 |
| Answer Relevance | 回答は質問に答えているか | RAGAS Answer Relevancy, LLM-as-a-judge |

### ツール・ベンチマーク

- **RAGAS**: LLM-as-a-judge によるRAG評価フレームワーク
- **BEIR / MTEB**: 検索器・埋め込みモデルの汎用ベンチマーク
- **HotpotQA / MuSiQue**: マルチホップQA
- **EnterpriseRAG-Bench**: 企業内部知識に特化(次節)

## 12. EnterpriseRAG-Bench

Onyx (2026) による**企業内部ドキュメントに特化したRAGベンチマーク** ([arXiv:2605.05253](https://arxiv.org/abs/2605.05253), [GitHub](https://github.com/onyx-dot-app/EnterpriseRAG-Bench))。

既存ベンチマークがWikipedia等の公開文書中心なのに対し,実際の社内データの特性──**乱雑さ,ノイズ,チャットログ/チケット/CRMレコードなどの文書タイプ,独自用語**──を再現している点が特徴。

- **コーパス**: 9種の企業ソース(Slack 275K, Gmail 120K, Linear 35K, Google Drive 25K, HubSpot 15K, Fireflies 10K, GitHub 8K, Jira 6K, Confluence 5K)からなる約50万件の合成文書。共通のプロジェクト・人物・イニシアチブに接地して文書間の整合性を持たせ,誤配置文書・ニアデュプリケート・矛盾情報などの現実的ノイズを注入している
- **質問**: 10カテゴリ500問。単一文書の単純検索 (basic),意味的検索 (semantic),文書内推論,制約付き検索 (constrained),矛盾解決 (conflicting_info),網羅性 (completeness),**情報が存在しないことの認識 (info_not_found)** など
- **評価**: 正解文書ID (`expected_doc_ids`) と正解回答 (`gold_answer`, `answer_facts`) が付与され,検索と回答生成の両方を評価できる

**本リポジトリでの再現**: デモアプリでは,このうち Confluence + Jira + GitHub の3ソース(計19,361文書)をサブセットとして使用し,この3ソースで完結する178問を検索対象クエリの例として利用できるようにした(`data/` 以下)。

## 13. 手法比較表

| 手法 | 改善する段階 | 長所 | 短所 | 追加コスト |
|---|---|---|---|---|
| Naive RAG (単一検索) | — | 単純・高速・安価 | 語彙/抽象度ミスマッチに弱い | なし |
| BM25 (疎) | Retrieval | 固有名詞・型番に強い,学習不要 | 同義語に弱い | なし |
| Dense (埋め込み) | Retrieval | 意味的類似に強い | ドメイン適合性依存 | 埋め込み計算 |
| Hybrid + RRF | Retrieval | 疎密の相互補完で安定 | インデックス2系統 | 小 |
| リランキング | Post-Retrieval | precision大幅向上 | 候補数比例のコスト | 中 |
| Query Rewriting | Pre-Retrieval | 簡単で効果安定 | 意図ずれリスク | LLM 1回 |
| HyDE | Pre-Retrieval | 抽象度ギャップ解消 | 誤仮説で誤誘導 | LLM 1回 |
| RAG-Fusion | Pre-Retrieval | recall向上,言い回し非依存 | 検索回数増 | LLM 1回+検索N回 |
| Small-to-Big | Indexing | 検索精度と文脈量の両立 | 親子管理が複雑 | 小 |
| Contextual Retrieval | Indexing | チャンクの文脈喪失を補う | 構築時LLMコスト | 構築時 大 |
| RAPTOR | Indexing | 抽象的質問に対応 | 構築・更新コスト大 | 構築時 大 |
| Self-RAG | 全体 | 検索要否・関連性を自己判断 | 専用学習が必要 | 中 |
| CRAG | Post-Retrieval | 検索失敗時のロバスト性 | 評価器精度に依存 | LLM 1〜2回 |
| Adaptive-RAG | Routing | コストと精度の最適化 | 分類器の調整 | 小〜中 |
| GraphRAG | Indexing/Retrieval | 全体俯瞰・マルチホップに強い | 構築コスト非常に大 | 構築時 特大 |
| Agentic RAG | 全体 | 最も柔軟,複雑質問に最強 | 高レイテンシ,非決定的 | LLM 多数回 |

## 14. デモアプリとの対応

本リポジトリの `app.py` (Streamlit) では,埋め込みAPI不要で動く範囲で以下を実装した。検索器は BM25 (rank_bm25) と TF-IDF 文字n-gram (scikit-learn) の2系統,生成は `claude-sonnet-5`。

| デモの手法 | 対応するSurvey節 | 実装内容 |
|---|---|---|
| Naive RAG (BM25) | §4.1 | BM25 top-k → 生成 |
| Naive RAG (TF-IDF) | §4.1 | TF-IDF文字n-gramコサイン top-k → 生成 |
| Hybrid + RRF | §4.3 | BM25 + TF-IDF を RRF 統合 → 生成 |
| Query Rewriting | §5.1 | LLMでクエリを検索向けに書き換え → Hybrid検索 |
| HyDE | §5.2 | LLMで仮想文書生成 → それで検索 |
| RAG-Fusion | §5.3 | LLMで派生クエリ4本生成 → 並列検索 → RRF |
| LLM Rerank | §4.5 | Hybrid で候補20件 → LLMが関連度採点 → 上位k |
| Corrective RAG (簡易版) | §7.2 | LLMが検索結果を評価 → 不十分ならクエリ改訂して再検索 |

※ Self-RAG は専用学習モデルが必要,GraphRAG はインデックス構築コストが大きいため,デモでは簡易版・代替(CRAG,RAG-Fusion)で代表させた。

## 15. 参考文献

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020) — RAGの原典
- Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey" (2023) — Naive/Advanced/Modular の分類
- Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering" (2020)
- Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search" (2020)
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022) — HyDE
- Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (2023)
- Yan et al., "Corrective Retrieval Augmented Generation" (2024) — CRAG
- Jeong et al., "Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity" (2024)
- Jiang et al., "Active Retrieval Augmented Generation" (2023) — FLARE
- Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (2024) — GraphRAG
- Sarthi et al., "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (2024)
- Anthropic, "Introducing Contextual Retrieval" (2024)
- Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023)
- Onyx, "EnterpriseRAG-Bench: A RAG Benchmark for Company Internal Knowledge" (2026) — [arXiv:2605.05253](https://arxiv.org/abs/2605.05253)
