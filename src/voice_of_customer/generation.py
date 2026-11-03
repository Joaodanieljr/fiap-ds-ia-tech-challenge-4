"""Geracao de respostas com Gemini, sempre condicionada a evidencias."""

from google import genai

from voice_of_customer.config import Settings
from voice_of_customer.models import Evidence


class GeminiAnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ValueError("Defina GEMINI_API_KEY no arquivo .env local.")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    def generate(self, question: str, evidence: list[Evidence]) -> str:
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
        return response.text or "Nao foi possivel gerar uma resposta com as evidencias recuperadas."
