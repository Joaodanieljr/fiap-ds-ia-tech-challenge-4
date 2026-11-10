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


def load_settings() -> Settings:
    load_dotenv()
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    if top_k < 1:
        raise ValueError("RETRIEVAL_TOP_K precisa ser maior que zero.")

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
    )
