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


def indexar_base(
    tamanho_amostra: int | None = None,
    persist_directory: str | Path = PASTA_INDICE,
    collection_name: str = NOME_COLECAO,
) -> None:
    base = preparar_base()
    if tamanho_amostra:
        base = base.sample(tamanho_amostra, random_state=42)   # amostra reprodutível
    print(f"Documentos a indexar: {len(base)}")

    # Vetores normalizados: a similaridade de cosseno fica direta e estável
    modelo = SentenceTransformer(MODELO_EMBEDDING)
    embeddings = modelo.encode(
        base["texto"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)
    cliente = chromadb.PersistentClient(path=str(persist_directory))
    # Recria a coleção do zero: rodar o script duas vezes não duplica nem mistura amostra com base completa
    try:
        cliente.delete_collection(collection_name)
    except Exception:
        pass   # coleção ainda não existe na primeira execução
    colecao = cliente.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )

    for inicio in range(0, len(base), TAMANHO_LOTE):
        fim = inicio + TAMANHO_LOTE
        lote = base.iloc[inicio:fim]
        colecao.add(
            ids=lote["review_id"].tolist(),
            documents=lote["texto"].tolist(),
            embeddings=embeddings[inicio:fim].tolist(),
            metadatas=lote[COLUNAS_METADADOS].to_dict("records"),
        )

    print(
        f"Índice gravado em {persist_directory} / {collection_name} "
        f"({colecao.count()} documentos)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa as avaliações da Olist no ChromaDB.")
    parser.add_argument("--amostra", type=int, help="indexa só N avaliações aleatórias")
    parser.add_argument(
        "--persist-directory",
        default=str(PASTA_INDICE),
        help="diretório persistente do Chroma (padrão: data/chroma)",
    )
    parser.add_argument(
        "--collection-name",
        default=NOME_COLECAO,
        help=f"coleção Chroma (padrão: {NOME_COLECAO})",
    )
    argumentos = parser.parse_args()
    indexar_base(
        tamanho_amostra=argumentos.amostra,
        persist_directory=argumentos.persist_directory,
        collection_name=argumentos.collection_name,
    )