"""
Prepara a base de conhecimento: limpa os comentários das avaliações e junta
os metadados usados nos filtros do RAG (nota, categoria, data, entrega).
Regras de limpeza justificadas na seção "Base de conhecimento" do README.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_DADOS = RAIZ_PROJETO / "data" / "raw"

# Abaixo disso são quase só elogios genéricos ("Bom", "Ok", ".")
TAMANHO_MINIMO_TEXTO = 10


def _montar_texto(avaliacoes: pd.DataFrame) -> pd.Series:
    """Junta título e mensagem e normaliza espaços e quebras de linha."""
    titulo = avaliacoes["review_comment_title"].fillna("").str.strip()
    mensagem = avaliacoes["review_comment_message"].fillna("").str.strip()
    texto = (titulo + ". " + mensagem).str.strip(" .")
    return texto.str.replace(r"\s+", " ", regex=True)


def _metadados_pedido(pedidos: pd.DataFrame) -> pd.DataFrame:
    """Data da compra e situação da entrega (comparada à data prevista)."""
    compra = pd.to_datetime(pedidos["order_purchase_timestamp"])
    entregue = pd.to_datetime(pedidos["order_delivered_customer_date"]).dt.normalize()
    prevista = pd.to_datetime(pedidos["order_estimated_delivery_date"]).dt.normalize()

    return pd.DataFrame({
        "order_id": pedidos["order_id"],
        "data_compra": compra.dt.strftime("%Y-%m-%d"),          # para exibição
        "ano_mes": compra.dt.year * 100 + compra.dt.month,      # para filtro por período (ex.: 201803)
        # "nao_entregue" inclui pedidos sem data de entrega registrada
        "situacao_entrega": np.select(
            [entregue.isna(), entregue > prevista],
            ["nao_entregue", "atrasado"],
            default="no_prazo",
        ),
    })


def _categoria_principal(itens: pd.DataFrame, produtos: pd.DataFrame) -> pd.DataFrame:
    """Uma categoria por pedido: a do item de maior preço (~0,7% têm mais de uma)."""
    itens_categoria = itens.merge(
        produtos[["product_id", "product_category_name"]], on="product_id", how="left"
    )
    return (
        itens_categoria.sort_values("price", ascending=False)
        .drop_duplicates("order_id")
        .rename(columns={"product_category_name": "categoria"})[["order_id", "categoria"]]
    )


def preparar_base() -> pd.DataFrame:
    avaliacoes = pd.read_csv(PASTA_DADOS / "olist_order_reviews_dataset.csv")
    pedidos = pd.read_csv(PASTA_DADOS / "olist_orders_dataset.csv")
    itens = pd.read_csv(PASTA_DADOS / "olist_order_items_dataset.csv")
    produtos = pd.read_csv(PASTA_DADOS / "olist_products_dataset.csv")

    # Mesmo review replicado em pedidos comprados juntos
    avaliacoes = avaliacoes.drop_duplicates("review_id")

    avaliacoes["texto"] = _montar_texto(avaliacoes)
    avaliacoes = avaliacoes[avaliacoes["texto"].str.len() >= TAMANHO_MINIMO_TEXTO]

    base = (
        avaliacoes[["review_id", "order_id", "review_score", "texto"]]
        .merge(_metadados_pedido(pedidos), on="order_id", how="left")
        .merge(_categoria_principal(itens, produtos), on="order_id", how="left")
        .rename(columns={"review_score": "nota"})
    )
    base["categoria"] = base["categoria"].fillna("sem_categoria")

    # Garantias para o Chroma: ID único e nenhum metadado nulo
    assert base["review_id"].is_unique, "review_id duplicado após o join"
    assert base.notna().all().all(), "há valores nulos na base"
    return base