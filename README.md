# FIAP Data Science com IA: Tech Challenge Fase 4

Solucao de Voice of Customer com RAG sobre avaliacoes do Brazilian E-Commerce Public Dataset by Olist.

## Estado

Scaffold em desenvolvimento. A branch `feat/base-conhecimento` define inicialmente o indice Chroma em `data/chroma`, colecao `avaliacoes_olist`, IDs `review_id`, metadados `nota`, `categoria`, `data_compra`, `ano_mes` e `situacao_entrega`, e embeddings `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` normalizados com distancia cosine. A compatibilidade e o acesso ao indice ainda precisam ser validados com uma consulta real. Consulte [plan.md](plan.md) para requisitos, cronograma, decisoes pendentes e plano de integracao.

## Desenvolvimento local

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
pytest
streamlit run app.py
```

Configure `GEMINI_API_KEY` no `.env`. Ajuste `CHROMA_PERSIST_DIRECTORY` e `CHROMA_COLLECTION_NAME` somente se o indice recebido usar outro caminho ou nome. Nunca versione o `.env`.
