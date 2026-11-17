"""Consulta ao indice Chroma usando o embedding definido na indexacao."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from voice_of_customer.models import Evidence

if TYPE_CHECKING:
    from voice_of_customer.config import Settings


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingVector(Protocol):
    def tolist(self) -> list[float]: ...


class TextEmbeddingModel(Protocol):
    def encode(
        self, sentences: str, *, normalize_embeddings: bool = False
    ) -> EmbeddingVector: ...


class ReviewRetriever(Protocol):
    def retrieve(
        self,
        question: str,
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> list[Evidence]: ...


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str) -> TextEmbeddingModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class ChromaReviewRetriever:
    def __init__(self, settings: Settings) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        self.collection = client.get_collection(settings.chroma_collection_name)
        self.embedding_model = _load_embedding_model(EMBEDDING_MODEL)

    def retrieve(
        self,
        question: str,
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> list[Evidence]:
        query_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        evidence: list[Evidence] = []
        for index, document_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else None
            metadata = metadata or {}
            text = documents[index] if index < len(documents) else None
            if not text:
                continue
            distance = distances[index] if index < len(distances) else None
            review_id = metadata.get("review_id", document_id)
            evidence.append(
                Evidence(
                    review_id=str(review_id),
                    text=str(text),
                    metadata=metadata,
                    distance=float(distance) if distance is not None else None,
                )
            )
        return evidence
