"""Orquestracao RAG com abstencao quando a busca nao retorna evidencias."""

from typing import Protocol

from voice_of_customer.models import Answer, Evidence
from voice_of_customer.retrieval import ReviewRetriever


class AnswerGenerator(Protocol):
    def generate(self, question: str, evidence: list[Evidence]) -> str: ...


class RagPipeline:
    def __init__(
        self,
        retriever: ReviewRetriever,
        generator: AnswerGenerator,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(
        self,
        question: str,
        where: dict[str, object] | None = None,
    ) -> Answer:
        evidence = self.retriever.retrieve(
            question=question,
            limit=self.top_k,
            where=where,
        )
        if not evidence:
            return Answer(
                text="Nao encontrei avaliacoes relevantes para responder com seguranca.",
                evidence=[],
                abstained=True,
            )

        return Answer(
            text=self.generator.generate(question, evidence),
            evidence=evidence,
        )
