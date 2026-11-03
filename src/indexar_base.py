"""
Gera os embeddings das avaliações e grava o índice persistido no ChromaDB.

Uso (na raiz do repo):
    python src/indexar_base.py --amostra 2000   # índice rápido para desenvolvimento
    python src/indexar_base.py                  # base completa
"""
import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from preparar_base import preparar_base

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_INDICE = RAIZ_PROJETO / "data" / "chroma"   # já coberta pelo .gitignore (data/)

# Contrato do índice combinado com a frente do pipeline RAG
NOME_COLECAO = "avaliacoes_olist"
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLUNAS_METADADOS = ["nota", "categoria", "data_compra", "ano_mes", "situacao_entrega"]

TAMANHO_LOTE = 1000   # o Chroma limita quantos registros aceita por chamada


def indexar_base(tamanho_amostra: int | None = None) -> None:
    base = preparar_base()
    if tamanho_amostra:
        base = base.sample(tamanho_amostra, random_state=42)   # amostra reprodutível
    print(f"Documentos a indexar: {len(base)}")

    # Vetores normalizados: a similaridade de cosseno fica direta e estável
    modelo = SentenceTransformer(MODELO_EMBEDDING)
    embeddings = modelo.encode(
        base["texto"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    cliente = chromadb.PersistentClient(path=str(PASTA_INDICE))
    # Recria a coleção do zero: rodar o script duas vezes não duplica nem mistura amostra com base completa
    try:
        cliente.delete_collection(NOME_COLECAO)
    except Exception:
        pass   # coleção ainda não existe na primeira execução
    colecao = cliente.create_collection(NOME_COLECAO, metadata={"hnsw:space": "cosine"})

    for inicio in range(0, len(base), TAMANHO_LOTE):
        fim = inicio + TAMANHO_LOTE
        lote = base.iloc[inicio:fim]
        colecao.add(
            ids=lote["review_id"].tolist(),
            documents=lote["texto"].tolist(),
            embeddings=embeddings[inicio:fim].tolist(),
            metadatas=lote[COLUNAS_METADADOS].to_dict("records"),
        )

    print(f"Índice gravado em {PASTA_INDICE} ({colecao.count()} documentos)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa as avaliações da Olist no ChromaDB.")
    parser.add_argument("--amostra", type=int, help="indexa só N avaliações aleatórias")
    indexar_base(parser.parse_args().amostra)