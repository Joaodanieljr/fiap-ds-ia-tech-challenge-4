"""Geracao de respostas com Gemini, sempre condicionada a evidencias."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from voice_of_customer.models import Evidence

if TYPE_CHECKING:
    from voice_of_customer.config import Settings


class GeminiAnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model
        self.client: Any | None = None

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        if not evidence:
            raise ValueError("GeminiAnswerGenerator requer evidencias para responder.")
        if not self.api_key:
            raise ValueError("Defina GEMINI_API_KEY no arquivo .env local.")

        if self.client is None:
            from google import genai

            self.client = genai.Client(api_key=self.api_key)

        context = "\n\n".join(
            f"[review_id={item.review_id}] {item.text}" for item in evidence
        )
        prompt = f"""Voce e um analista de Voice of Customer. Responda em portugues brasileiro usando somente as avaliacoes fornecidas.

Regras:
- Nao invente fatos, frequencias, causas ou exemplos que nao estejam sustentados pelo contexto.
- Diferencie observacao qualitativa de uma conclusao quantitativa.
- Cite os reviews que sustentam cada ponto usando exatamente [review_id=...].
- Se as evidencias nao forem suficientes para responder, diga isso claramente.

Pergunta: {question}

Avaliacoes recuperadas:
{context}
"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        if not response.text:
            raise RuntimeError("O Gemini retornou uma resposta vazia.")
        return response.text
