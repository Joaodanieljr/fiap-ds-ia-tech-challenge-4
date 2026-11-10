"""Configuracao carregada do ambiente local."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    chroma_persist_directory: str
    chroma_collection_name: str
    retrieval_top_k: int
    retrieval_max_distance: float | None = None


def load_settings() -> Settings:
    load_dotenv()
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    if top_k < 1:
        raise ValueError("RETRIEVAL_TOP_K precisa ser maior que zero.")

    max_distance_value = os.getenv("RETRIEVAL_MAX_DISTANCE", "").strip()
    max_distance = float(max_distance_value) if max_distance_value else None
    if max_distance is not None and not 0 <= max_distance <= 2:
        raise ValueError("RETRIEVAL_MAX_DISTANCE precisa estar entre 0 e 2.")

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        chroma_persist_directory=os.getenv(
            "CHROMA_PERSIST_DIRECTORY", "./data/chroma"
        ),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME", "avaliacoes_olist"
        ),
        retrieval_top_k=top_k,
        retrieval_max_distance=max_distance,
    )
