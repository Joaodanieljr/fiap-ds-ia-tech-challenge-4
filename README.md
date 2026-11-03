# fiap-ds-ia-tech-challenge-4
Quarto tech challenge do curso de Data Science com IA

## Instalação e execução

### Pré-requisitos

- Python 3.11 ou 3.12
- Git

### 1. Clonar o repositório e criar o ambiente virtual

```bash
git clone <URL_DO_REPO>
cd <nome-do-repo>
python -m venv .venv
```

Ative o ambiente:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Baixar os dados

```bash
python src/baixar_dados.py
```

O script baixa o [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
via `kagglehub` e copia para `data/raw/` apenas as tabelas usadas pela solução:

| Arquivo | Uso na solução |
|---|---|
| `olist_order_reviews_dataset.csv` | Comentários das avaliações (texto da base de conhecimento) e nota |
| `olist_orders_dataset.csv` | Datas de compra e entrega (cálculo de atraso) |
| `olist_order_items_dataset.csv` | Relação entre pedido e produto |
| `olist_products_dataset.csv` | Categoria do produto |

> Os dados não são versionados no Git (`data/` está no `.gitignore`). O dataset é público e
> estático (pedidos de 2016 a 2018), então qualquer execução obtém a mesma base.
>
> Caso o `kagglehub` solicite autenticação, gere um token em *kaggle.com → Settings → API →
> Create New Token* e salve o `kaggle.json` em `~/.kaggle/` (Windows: `C:\Users\<usuario>\.kaggle\`).