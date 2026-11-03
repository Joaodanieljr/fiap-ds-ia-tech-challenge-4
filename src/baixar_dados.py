"""
Baixa o dataset público da Olist (Kaggle) e copia as tabelas
usadas pela base de conhecimento para data/raw/.

Uso (na raiz do repo):  python src/baixar_dados.py
"""
import shutil
from pathlib import Path

import kagglehub

# Raiz do repo calculada a partir deste arquivo: funciona de qualquer pasta
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_DESTINO = RAIZ_PROJETO / "data" / "raw"

# Só as tabelas necessárias (YAGNI: o dataset tem 9 CSVs)
ARQUIVOS_NECESSARIOS = [
    "olist_order_reviews_dataset.csv",   # comentários (texto principal)
    "olist_orders_dataset.csv",          # datas de compra e entrega
    "olist_order_items_dataset.csv",     # liga pedido ao produto
    "olist_products_dataset.csv",        # categoria do produto
]


def baixar_dados() -> None:
    # Baixa do Kaggle ou reaproveita o cache local
    pasta_cache = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    PASTA_DESTINO.mkdir(parents=True, exist_ok=True)

    for nome_arquivo in ARQUIVOS_NECESSARIOS:
        shutil.copy2(pasta_cache / nome_arquivo, PASTA_DESTINO / nome_arquivo)
        print(f"ok  {nome_arquivo}")

    print(f"\nArquivos em: {PASTA_DESTINO}")


if __name__ == "__main__":
    baixar_dados()