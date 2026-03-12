"""
ingest.py - PDF ingestion pipeline for KSSM Math textbooks
Extracts text using PyMuPDF, falls back to EasyOCR for image-heavy pages,
chunks text, generates embeddings, and uploads to Zilliz Cloud.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF
import easyocr
import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import config

logger = logging.getLogger(__name__)

# Global EasyOCR reader (loaded once to save memory)
_ocr_reader = None


def get_ocr_reader() -> easyocr.Reader:
    """Load EasyOCR reader (supports Malay + English)."""
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("🔤 Loading EasyOCR (ms + en)... This may take a moment.")
        _ocr_reader = easyocr.Reader(["ms", "en"], gpu=True)
        logger.info("✅ EasyOCR loaded.")
    return _ocr_reader


def detect_form_level(filename: str) -> str:
    """Extract form level from filename like T1_Math_BukuTeks_KSSM.pdf."""
    basename = Path(filename).stem.upper()
    for key, value in config.FORM_LEVEL_MAP.items():
        if basename.startswith(key):
            return value
    return "Unknown"


def extract_text_from_page(page: fitz.Page) -> Tuple[str, bool]:
    """
    Extract text from a PDF page.
    Returns (text, used_ocr) tuple.
    - First tries PyMuPDF native text extraction
    - Falls back to EasyOCR if page appears image-based
    """
    text = page.get_text("text").strip()

    # Heuristic: if native text is too short, use OCR
    if len(text) < 50:
        try:
            reader = get_ocr_reader()
            # Render page as high-res image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR accuracy
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            results = reader.readtext(img_array, detail=0, paragraph=True)
            ocr_text = "\n".join(results).strip()
            if ocr_text:
                return ocr_text, True
        except Exception as e:
            logger.warning(f"⚠️  OCR failed on page: {e}")

    return text, False


def extract_pdf_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract all text chunks from a KSSM PDF with metadata.

    Returns list of dicts:
        {text, source_file, form_level, page_num, chunk_idx}
    """
    path = Path(pdf_path)
    form_level = detect_form_level(path.name)
    source_file = path.name

    logger.info(f"\n📚 Processing: {source_file} ({form_level})")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    doc = fitz.open(str(path))
    all_chunks = []
    ocr_page_count = 0

    for page_idx in tqdm(range(len(doc)), desc=f"  Pages ({source_file})"):
        page = doc[page_idx]
        page_num = page_idx + 1

        text, used_ocr = extract_text_from_page(page)
        if used_ocr:
            ocr_page_count += 1

        if not text:
            continue

        # Clean up text
        text = re.sub(r"\s+", " ", text).strip()

        # Split into chunks
        chunks = text_splitter.split_text(text)

        for chunk_idx, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 20:  # skip very short chunks
                continue
            all_chunks.append(
                {
                    "text": chunk_text,
                    "source_file": source_file,
                    "form_level": form_level,
                    "page_num": page_num,
                    "chunk_idx": chunk_idx,
                }
            )

    doc.close()
    logger.info(
        f"  ✅ Extracted {len(all_chunks)} chunks | OCR pages: {ocr_page_count}"
    )
    return all_chunks


def generate_embeddings(
    chunks: List[Dict[str, Any]], model: SentenceTransformer, batch_size: int = 32
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for all chunks using sentence-transformers.
    Returns chunks with 'embedding' key added.
    """
    texts = [c["text"] for c in chunks]
    logger.info(f"  🧮 Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # for cosine similarity
    )

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    return chunks


def find_pdf_files(pdf_dir: str) -> List[str]:
    """Find all KSSM Math PDF files in the specified directory."""
    pdf_dir_path = Path(pdf_dir)
    patterns = ["T*_Math_BukuTeks_KSSM.pdf", "T*_Math_*.pdf", "*.pdf"]

    for pattern in patterns:
        files = sorted(pdf_dir_path.glob(pattern))
        if files:
            logger.info(f"Found {len(files)} PDF(s) using pattern '{pattern}'")
            return [str(f) for f in files]

    return []


def run_ingestion(
    pdf_dir: str = None,
    collection=None,
    dry_run: bool = False,
    batch_insert_size: int = 100,
) -> Dict[str, Any]:
    """
    Main ingestion pipeline.

    Args:
        pdf_dir: Directory containing PDFs (defaults to config)
        collection: Milvus collection (if None, initializes one)
        dry_run: If True, processes PDFs but does not upload to Zilliz
        batch_insert_size: Number of chunks per Zilliz insert batch

    Returns:
        Summary statistics dict
    """
    from backend.vector_store import init_collection, upsert_chunks

    if pdf_dir is None:
        pdf_dir = config.PDF_DIR

    pdf_files = find_pdf_files(pdf_dir)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    logger.info(f"📂 Found {len(pdf_files)} PDFs to ingest:")
    for f in pdf_files:
        logger.info(f"   - {Path(f).name}")

    # Load embedding model
    logger.info(f"\n🔗 Loading embedding model: {config.EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    logger.info("✅ Embedding model loaded.")

    # Initialize Zilliz collection
    if not dry_run:
        if collection is None:
            collection = init_collection()

    stats = {
        "total_chunks": 0,
        "total_pages": 0,
        "files_processed": [],
        "dry_run": dry_run,
    }

    for pdf_path in pdf_files:
        # Extract chunks
        chunks = extract_pdf_chunks(pdf_path)
        if not chunks:
            logger.warning(f"⚠️  No chunks extracted from {pdf_path}")
            continue

        # Generate embeddings
        chunks = generate_embeddings(chunks, embed_model)

        # Batch insert into Zilliz
        if not dry_run:
            logger.info(f"  ⬆️  Uploading {len(chunks)} chunks to Zilliz Cloud...")
            for i in range(0, len(chunks), batch_insert_size):
                batch = chunks[i : i + batch_insert_size]
                upsert_chunks(batch, collection)
                logger.info(
                    f"    Uploaded batch {i // batch_insert_size + 1} "
                    f"({len(batch)} chunks)"
                )

        stats["total_chunks"] += len(chunks)
        stats["files_processed"].append(Path(pdf_path).name)

    logger.info(
        f"\n🎉 Ingestion complete!"
        f"\n   Files processed: {len(stats['files_processed'])}"
        f"\n   Total chunks: {stats['total_chunks']}"
        f"\n   Mode: {'DRY RUN (no upload)' if dry_run else 'UPLOADED to Zilliz'}"
    )
    return stats
