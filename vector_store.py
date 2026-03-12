"""
vector_store.py - Zilliz Cloud (Milvus) vector database operations
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from backend.config import config

logger = logging.getLogger(__name__)


def connect_zilliz():
    """Establish connection to Zilliz Cloud."""
    connections.connect(
        alias="default",
        uri=config.ZILLIZ_URI,
        token=config.ZILLIZ_TOKEN,
    )
    logger.info(f"✅ Connected to Zilliz Cloud: {config.ZILLIZ_URI}")


def init_collection() -> Collection:
    """Create collection if it doesn't exist, then return it."""
    connect_zilliz()

    if utility.has_collection(config.COLLECTION_NAME):
        logger.info(f"📦 Collection '{config.COLLECTION_NAME}' already exists.")
        collection = Collection(config.COLLECTION_NAME)
        collection.load()
        return collection

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="form_level", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="page_num", dtype=DataType.INT32),
        FieldSchema(name="chunk_idx", dtype=DataType.INT32),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=config.EMBEDDING_DIM,
        ),
    ]
    schema = CollectionSchema(fields, description="Malaysia KSSM Math Textbook Chunks")
    collection = Collection(name=config.COLLECTION_NAME, schema=schema)

    # Create IVF_FLAT index for fast similarity search
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()

    logger.info(f"✅ Created new collection '{config.COLLECTION_NAME}'")
    return collection


def upsert_chunks(chunks: List[Dict[str, Any]], collection: Collection):
    """
    Batch insert text chunks with embeddings into Zilliz.

    Args:
        chunks: List of dicts with keys: text, embedding, source_file, form_level, page_num, chunk_idx
        collection: Milvus Collection object
    """
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    source_files = [c["source_file"] for c in chunks]
    form_levels = [c["form_level"] for c in chunks]
    page_nums = [c["page_num"] for c in chunks]
    chunk_idxs = [c["chunk_idx"] for c in chunks]

    data = [
        texts,
        source_files,
        form_levels,
        page_nums,
        chunk_idxs,
        embeddings,
    ]

    collection.insert(data)
    collection.flush()
    logger.info(f"✅ Inserted {len(chunks)} chunks into Zilliz")


def similarity_search(
    query_embedding: List[float],
    collection: Collection,
    top_k: int = None,
    form_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks in the vector store.

    Args:
        query_embedding: The query vector
        collection: Milvus Collection object
        top_k: Number of results to return
        form_filter: Optional form level filter (e.g., "T1", "T2")

    Returns:
        List of dicts with text, source_file, form_level, page_num, score
    """
    if top_k is None:
        top_k = config.TOP_K

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}

    expr = None
    if form_filter:
        form_level_val = config.FORM_LEVEL_MAP.get(form_filter, "")
        if form_level_val:
            expr = f'form_level == "{form_level_val}"'

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["text", "source_file", "form_level", "page_num"],
    )

    hits = []
    for hit in results[0]:
        hits.append(
            {
                "text": hit.entity.get("text"),
                "source_file": hit.entity.get("source_file"),
                "form_level": hit.entity.get("form_level"),
                "page_num": hit.entity.get("page_num"),
                "score": hit.score,
            }
        )
    return hits


def get_collection_stats(collection: Collection) -> Dict[str, Any]:
    """Return basic stats about the collection."""
    return {
        "name": collection.name,
        "num_entities": collection.num_entities,
        "schema": str(collection.schema),
    }
