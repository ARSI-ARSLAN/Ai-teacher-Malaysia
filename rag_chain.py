"""
rag_chain.py - RAG pipeline for AI Teacher Malaysia
Orchestrates: query embedding → vector search → prompt building → Groq LLM
"""

import logging
from typing import Generator, List, Dict, Any, Optional

from langdetect import detect as langdetect_detect, LangDetectException
from sentence_transformers import SentenceTransformer

from backend.config import config
from backend.llm import get_llm

logger = logging.getLogger(__name__)

# Singleton embedding model
_embed_model = None


def get_embed_model() -> SentenceTransformer:
    """Load embedding model once and reuse."""
    global _embed_model
    if _embed_model is None:
        logger.info(f"🔗 Loading embedding model: {config.EMBEDDING_MODEL}")
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("✅ Embedding model ready.")
    return _embed_model


# System prompt for AI Guru Matematik
SYSTEM_PROMPT = """You are AI Guru Matematik, an expert AI math tutor trained on the Malaysian KSSM (Kurikulum Standard Sekolah Menengah) syllabus for Forms 1 to 5.

Your responsibilities:
1. Answer student questions accurately based on the KSSM Math syllabus content provided.
2. Always respond in the SAME LANGUAGE the student used (Bahasa Malaysia or English).
3. For mathematical problems, ALWAYS provide step-by-step solutions with clear numbered steps.
4. Use proper mathematical notation and be beginner-friendly.
5. If the question is in Bahasa Malaysia, your ENTIRE response must be in Bahasa Malaysia.
6. If information isn't in the provided context, say so honestly but still help using general math knowledge.
7. Format math equations clearly using plain text (e.g., "2x + 5 = 13" not LaTeX unless using standard notation).

Tone: Encouraging, patient, and supportive — like a great school teacher.

Reference context from the KSSM textbooks is provided below. Use it as your primary knowledge source.
"""

BM_SYSTEM_PROMPT = """Anda ialah AI Guru Matematik, seorang tutor matematik AI pakar yang dilatih berdasarkan sukatan pelajaran KSSM (Kurikulum Standard Sekolah Menengah) Malaysia untuk Tingkatan 1 hingga 5.

Tanggungjawab anda:
1. Jawab soalan pelajar dengan tepat berdasarkan kandungan sukatan pelajaran KSSM yang disediakan.
2. Sentiasa balas dalam BAHASA MALAYSIA kerana pelajar menggunakan Bahasa Malaysia.
3. Untuk soalan matematik, SENTIASA berikan penyelesaian langkah demi langkah yang jelas.
4. Gunakan notasi matematik yang betul dan mesra untuk pemula.
5. Format persamaan matematik dengan jelas.
6. Jika maklumat tidak ada dalam konteks yang diberikan, akui tetapi bantu menggunakan pengetahuan matematik umum.

Nada: Menggalakkan, sabar, dan menyokong — seperti guru sekolah yang baik.

Konteks rujukan dari buku teks KSSM disediakan di bawah. Gunakan sebagai sumber pengetahuan utama anda.
"""


def format_context(search_results: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context string for the prompt."""
    if not search_results:
        return "No specific textbook content found for this query."

    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(
            f"[Source {i}: {result['form_level']} | {result['source_file']} | Page {result['page_num']}]\n"
            f"{result['text']}"
        )
    return "\n\n".join(context_parts)


def detect_language(text: str) -> str:
    """
    Detect whether the question is in Bahasa Malaysia or English.
    Uses langdetect library + BM keyword fallback.
    Returns 'bm' or 'en'.
    """
    # BM-specific keywords as fast pre-check (common math question words)
    bm_keywords = [
        "apakah", "bagaimana", "selesaikan", "kirakan", "cari", "tunjukkan",
        "jelaskan", "berikan", "nilai", "persamaan", "nombor", "bentuk",
        "tingkatan", "langkah", "pengiraan", "hasil", "buktikan", "tentukan",
        "daripada", "dengan", "untuk", "adalah", "ialah", "iaitu", "atau",
        "dan", "jika", "maka", "diberi", "diberi", "hitungkan",
    ]
    text_lower = text.lower()
    bm_hit = sum(1 for w in bm_keywords if w in text_lower.split())
    if bm_hit >= 1:
        return "bm"

    # Use langdetect for everything else
    try:
        detected = langdetect_detect(text)
        # langdetect returns 'ms' for Malay, 'id' for Indonesian (similar to BM)
        if detected in ("ms", "id"):
            return "bm"
        return "en"
    except LangDetectException:
        return "en"  # default to English if detection fails


def query(
    question: str,
    collection,
    language: str = "auto",
    form_filter: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Main RAG query function — retrieves context and streams Grok response.

    Args:
        question: Student's question
        collection: Milvus collection object
        language: 'bm', 'en', or 'auto' (auto-detect)
        form_filter: Optional form level filter like 'T1', 'T2', etc.

    Yields:
        Streamed response tokens from Grok
    """
    from backend.vector_store import similarity_search

    # Language detection
    if language == "auto":
        language = detect_language(question)
    logger.info(f"🌐 Detected language: {language.upper()} | Question: {question[:60]}...")

    # Embed the question
    embed_model = get_embed_model()
    query_embedding = embed_model.encode(
        question, normalize_embeddings=True
    ).tolist()

    # Retrieve relevant chunks from Zilliz
    search_results = similarity_search(
        query_embedding=query_embedding,
        collection=collection,
        top_k=config.TOP_K,
        form_filter=form_filter,
    )
    logger.info(f"🔍 Retrieved {len(search_results)} context chunks")

    # Format context
    context = format_context(search_results)

    # Select system prompt based on language
    system_prompt = BM_SYSTEM_PROMPT if language == "bm" else SYSTEM_PROMPT

    # Explicit language instruction injected into the user message (double signal)
    if language == "bm":
        lang_instruction = (
            "[PENTING: Jawab SELURUH respons dalam BAHASA MALAYSIA sahaja. "
            "Jangan gunakan bahasa Inggeris langsung.]"
        )
    else:
        lang_instruction = (
            "[IMPORTANT: Reply in ENGLISH only. Do not use Bahasa Malaysia.]"
        )

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{lang_instruction}\n\n"
                f"KSSM Textbook Context:\n{context}\n\n"
                f"---\n\nStudent Question: {question}"
            ),
        },
    ]

    # Stream response from Groq
    llm = get_llm()
    yield from llm.chat(messages=messages, stream=True, temperature=0.3)


def query_sync(
    question: str,
    collection,
    language: str = "auto",
    form_filter: Optional[str] = None,
) -> str:
    """Non-streaming version of query."""
    return "".join(query(question, collection, language, form_filter))
