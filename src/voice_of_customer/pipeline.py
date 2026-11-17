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
        max_distance: float | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.max_distance = max_distance

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
        if self.max_distance is not None:
            evidence = [
                item
                for item in evidence
                if item.distance is not None and item.distance <= self.max_distance
            ]
        if not evidence:
            return Answer(
                text=(
                    "Nao encontrei evidencias com similaridade suficiente para "
                    "responder com seguranca."
                ),
                evidence=[],
                abstained=True,
            )

        return Answer(
            text=self.generator.generate(question, evidence),
            evidence=evidence,
        )
