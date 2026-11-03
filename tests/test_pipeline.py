import sys
from types import ModuleType, SimpleNamespace

from voice_of_customer import retrieval
from voice_of_customer.models import Evidence
from voice_of_customer.pipeline import RagPipeline


class FakeRetriever:
    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence
        self.calls: list[dict[str, object]] = []

    def retrieve(self, question: str, limit: int, where=None) -> list[Evidence]:
        self.calls.append({"question": question, "limit": limit, "where": where})
        return self.evidence


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        self.calls += 1
        return f"Resposta baseada em {len(evidence)} review(s)."


def test_pipeline_abstains_without_evidence() -> None:
    generator = FakeGenerator()
    pipeline = RagPipeline(FakeRetriever([]), generator)

    result = pipeline.answer("Pergunta fora do escopo")

    assert result.abstained is True
    assert result.evidence == []
    assert generator.calls == 0


def test_pipeline_returns_answer_and_evidence() -> None:
    evidence = [Evidence(review_id="review-123", text="Entrega chegou atrasada.")]
    retriever = FakeRetriever(evidence)
    generator = FakeGenerator()
    pipeline = RagPipeline(retriever, generator, top_k=3)

    result = pipeline.answer("Houve problema com entrega?")

    assert result.abstained is False
    assert result.text == "Resposta baseada em 1 review(s)."
    assert result.evidence == evidence
    assert retriever.calls[0]["limit"] == 3
    assert generator.calls == 1


def test_chroma_retriever_uses_index_embedding_contract(monkeypatch) -> None:
    query_result = {
        "ids": [["review-123"]],
        "documents": [["Entrega chegou atrasada."]],
        "metadatas": [[{"nota": 1}]],
        "distances": [[0.2]],
    }

    class FakeVector:
        def tolist(self) -> list[float]:
            return [0.1, 0.2]

    class FakeEmbeddingModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def encode(
            self, sentences: str, *, normalize_embeddings: bool = False
        ) -> FakeVector:
            self.calls.append(
                {"sentences": sentences, "normalize_embeddings": normalize_embeddings}
            )
            return FakeVector()

    class FakeCollection:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def query(self, **kwargs):
            self.calls.append(kwargs)
            return query_result

    collection = FakeCollection()
    model = FakeEmbeddingModel()
    requested_collections: list[str] = []
    client = SimpleNamespace(
        get_collection=lambda name: requested_collections.append(name) or collection
    )
    chromadb = ModuleType("chromadb")
    chromadb.PersistentClient = lambda path: client
    monkeypatch.setitem(sys.modules, "chromadb", chromadb)
    loaded_models: list[str] = []

    def load_model(model_name: str) -> FakeEmbeddingModel:
        loaded_models.append(model_name)
        return model

    monkeypatch.setattr(retrieval, "_load_embedding_model", load_model)
    settings = SimpleNamespace(
        gemini_api_key="",
        gemini_model="gemini-test",
        chroma_persist_directory="data/chroma",
        chroma_collection_name="avaliacoes_olist",
        retrieval_top_k=5,
    )
    retriever = retrieval.ChromaReviewRetriever(settings)
    evidence = retriever.retrieve(
        "Houve problema com entrega?",
        limit=3,
        where={"nota": {"$lte": 2}},
    )

    assert requested_collections == ["avaliacoes_olist"]
    assert loaded_models == [retrieval.EMBEDDING_MODEL]
    assert model.calls == [
        {
            "sentences": "Houve problema com entrega?",
            "normalize_embeddings": True,
        }
    ]
    assert collection.calls == [
        {
            "query_embeddings": [[0.1, 0.2]],
            "n_results": 3,
            "where": {"nota": {"$lte": 2}},
            "include": ["documents", "metadatas", "distances"],
        }
    ]
    assert evidence == [
        Evidence(
            review_id="review-123",
            text="Entrega chegou atrasada.",
            metadata={"nota": 1},
            distance=0.2,
        )
    ]
