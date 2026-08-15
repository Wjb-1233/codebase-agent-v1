"""RAG 输入层：代码切块、embedding 和向量存储。"""

from .chunker import Chunk, chunk_file
from .embeddings import OpenAIEmbeddingProvider, SentenceTransformerEmbeddingProvider
from .vector_store import InMemoryVectorStore, SearchResult, search_code, cosine_similarity
from .qdrant_store import QdrantVectorStore, VectorStoreError
from .keyword_search import KeywordIndex, keyword_search
from .hybrid_search import rrf_fuse
from .reranker import Reranker, IdentityReranker, CrossEncoderReranker
from .parent_document import ParentDocument, attach_parent_metadata, build_parent_map, expand_with_parents

__all__ = [
    "Chunk",
    "chunk_file",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "VectorStoreError",
    "SearchResult",
    "search_code",
    "cosine_similarity",
    "KeywordIndex",
    "keyword_search",
    "rrf_fuse",
    "Reranker",
    "IdentityReranker",
    "CrossEncoderReranker",
    "ParentDocument",
    "attach_parent_metadata",
    "build_parent_map",
    "expand_with_parents",
]
