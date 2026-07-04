"""Allganize RAG-Evaluation-Dataset-JA を app 用 parquet に変換するスクリプト.

https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-JA
- documents.csv の URL から PDF 65本をダウンロードし, pypdf で1ページ=1文書として本文抽出
- テキスト層が無い/フォントのCIDマッピングが壊れているPDFは RapidOCR (日本語) で抽出
- rag_evaluation_result.csv の300問を questions parquet に変換
  (doc_id は "ファイル名#pページ番号" で正解ページと突き合わせる)

出力: data/documents_allganize_ja.parquet, data/questions_allganize_ja.parquet
"""

from __future__ import annotations

import io
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from pypdf import PdfReader

_CJK_RE = re.compile(r"[぀-ヿ一-鿿]")
MIN_PAGE_CHARS = 20   # これ未満のページはテキスト層なしとみなしOCRする
MIN_CJK_RATIO = 0.05  # 日本語文書なのにCJK比率がこれ未満なら文字化けとみなす

HF_BASE = "https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-JA/resolve/main"
DATA_DIR = Path(__file__).parent / "data"
PDF_CACHE = DATA_DIR / "allganize_ja_pdfs"
HEADERS = {
    # 官公庁サイトの一部は非ブラウザUAを403で弾くため,ブラウザ相当のUAを名乗る
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Accept-Language": "ja,en;q=0.8",
}


def fetch_pdf(row: pd.Series) -> tuple[str, bytes | None]:
    cache = PDF_CACHE / row["file_name"]
    if cache.exists():
        return row["file_name"], cache.read_bytes()
    try:
        resp = requests.get(row["url"], headers=HEADERS, timeout=60)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
        return row["file_name"], resp.content
    except Exception as e:  # noqa: BLE001 - 失敗したPDFはスキップして続行
        print(f"  !! {row['file_name']} のダウンロード失敗: {e}", file=sys.stderr)
        return row["file_name"], None


def extract_pages(file_name: str, title: str, domain: str,
                  raw: bytes) -> list[dict]:
    rows = []
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            reader.decrypt("")  # 閲覧パスワードなしの暗号化PDF
        n = len(reader.pages)
    except Exception as e:  # noqa: BLE001
        print(f"  !! {file_name} のPDF解析失敗: {e}", file=sys.stderr)
        return rows
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - 壊れたページは空文字扱い
            text = ""
        rows.append({
            "doc_id": f"{file_name}#p{i}",
            "source_type": domain,
            "title": f"{title} (p.{i}/{n})",
            "content": text.strip(),
        })
    return rows


_ocr_engine = None


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        import numpy  # noqa: F401 - rapidocr が必要とする
        from rapidocr import RapidOCR
        _ocr_engine = RapidOCR(
            params={"Rec.lang_type": "japan", "Global.log_level": "error"})
    return _ocr_engine


def _ocr_page(pdf, index: int) -> str:
    import numpy as np
    img = pdf[index].render(scale=2.0).to_pil()
    res = _get_ocr()(np.array(img))
    return "\n".join(res.txts) if res.txts else ""


def _is_garbled(pages: list[dict]) -> bool:
    """日本語文書としてCJK文字がほぼ無い = テキスト層が壊れているか画像のみ."""
    text = "".join(p["content"] for p in pages)
    return len(_CJK_RE.findall(text)) < len(text) * MIN_CJK_RATIO


def ocr_fallback(file_name: str, raw: bytes, pages: list[dict]) -> list[dict]:
    """壊れた文書は全ページ,正常な文書でも空のページのみOCRで抽出し直す."""
    import pypdfium2 as pdfium
    targets = (range(len(pages)) if _is_garbled(pages) else
               [i for i, p in enumerate(pages) if len(p["content"]) < MIN_PAGE_CHARS])
    if not targets:
        return pages
    print(f"  OCR: {file_name} ({len(targets)}ページ)")
    pdf = pdfium.PdfDocument(io.BytesIO(raw))
    for i in targets:
        try:
            pages[i]["content"] = _ocr_page(pdf, i).strip()
        except Exception as e:  # noqa: BLE001 - OCR失敗ページは元のまま残す
            print(f"  !! {file_name} p{i + 1} のOCR失敗: {e}", file=sys.stderr)
    return pages


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    PDF_CACHE.mkdir(exist_ok=True)

    docs_meta = pd.read_csv(f"{HF_BASE}/documents.csv")
    qa = pd.read_csv(f"{HF_BASE}/rag_evaluation_result.csv")
    # documents.csv 側のファイル名は NFD (濁点分解形) が混じるため NFC に揃える
    docs_meta["file_name"] = docs_meta["file_name"].map(
        lambda s: unicodedata.normalize("NFC", s))
    qa["target_file_name"] = qa["target_file_name"].map(
        lambda s: unicodedata.normalize("NFC", s))
    print(f"PDF {len(docs_meta)}本をダウンロード中 (キャッシュ: {PDF_CACHE}) ...")

    with ThreadPoolExecutor(max_workers=8) as ex:
        blobs = dict(ex.map(fetch_pdf, (r for _, r in docs_meta.iterrows())))

    doc_rows: list[dict] = []
    for _, r in docs_meta.iterrows():
        raw = blobs.get(r["file_name"])
        if raw is None:
            continue
        pages = extract_pages(r["file_name"], r["title"], r["domain"], raw)
        doc_rows.extend(ocr_fallback(r["file_name"], raw, pages))
    docs = pd.DataFrame(doc_rows).astype("string")
    empty = (docs["content"].str.len() == 0).sum()
    print(f"文書 {len(docs)}ページを抽出 (本文が空のページ: {empty})")

    valid_ids = set(docs["doc_id"])
    questions = pd.DataFrame({
        "question_id": [f"ja_{i:04d}" for i in range(1, len(qa) + 1)],
        "question_type": qa["type"],
        "source_types": qa["domain"].map(lambda d: [d]),
        "question": qa["question"],
        "expected_doc_ids": [
            [f"{f}#p{int(p)}"]
            for f, p in zip(qa["target_file_name"], qa["target_page_no"])
        ],
        "gold_answer": qa["target_answer"],
    })
    # ダウンロードに失敗したPDFに紐づく質問は除外
    ok = questions["expected_doc_ids"].map(lambda ids: all(i in valid_ids for i in ids))
    if (~ok).any():
        print(f"正解文書が欠落している質問 {(~ok).sum()}問を除外")
        questions = questions[ok].reset_index(drop=True)

    docs.to_parquet(DATA_DIR / "documents_allganize_ja.parquet", index=False)
    questions.to_parquet(DATA_DIR / "questions_allganize_ja.parquet", index=False)
    print(f"保存完了: 文書 {len(docs)}件 / 質問 {len(questions)}問")


if __name__ == "__main__":
    main()
