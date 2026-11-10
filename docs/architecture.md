# Arquitetura e decisões técnicas

## Objetivo de negócio

Permitir que equipes explorem avaliações de clientes em linguagem natural e encontrem exemplos
concretos que apoiem uma análise de Voice of Customer. A solução prioriza rastreabilidade e suporte
factual; respostas geradas não devem ser interpretadas como estatística populacional.

## Fluxo de consulta

1. A interface recebe uma pergunta.
2. O mesmo modelo local usado na indexação transforma a pergunta em embedding normalizado.
3. Chroma recupera os reviews mais próximos por distância cosine.
4. O pipeline envia ao Gemini somente a pergunta e os reviews recuperados, identificados por
   `review_id`.
5. A interface apresenta a resposta, as evidências, os metadados e as distâncias retornadas.

## Fluxo de preparação

1. Baixar as tabelas públicas necessárias do dataset Olist.
2. Limpar e combinar título e comentário dos reviews; enriquecer os documentos com metadados de
   nota, categoria, data e situação da entrega.
3. Gerar embeddings com `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
4. Armazenar textos, IDs, embeddings e metadados em uma coleção Chroma persistida localmente.

As regras e estatísticas de preparação estão em [data-preparation.md](data-preparation.md).

## Componentes

| Componente | Responsabilidade |
|---|---|
| `app.py` | Interface Streamlit e composição dos adaptadores |
| `config.py` | Configuração local por variáveis de ambiente e `.env` |
| `retrieval.py` | Embedding da consulta e busca semântica no Chroma |
| `generation.py` | Prompt condicionado às evidências e chamada Gemini |
| `pipeline.py` | Orquestração entre recuperação, abstinência sem resultado e geração |
| `models.py` | Modelos de evidência e resposta |
| `preparar_base.py` | Limpeza dos textos e preparação de metadados |
| `indexar_base.py` | Geração e persistência do índice vetorial |

## Tecnologias

- **Python:** integração e lógica de aplicação.
- **Sentence Transformers:** embeddings locais multilíngues; sem custo de embedding por chamada.
- **ChromaDB:** busca vetorial persistida localmente e filtros por metadados.
- **Google Gemini:** síntese das evidências recuperadas em linguagem natural.
- **Streamlit:** interface de demonstração.
- **pytest:** testes unitários sem depender de serviços externos.

## Rastreabilidade e limites atuais

- Cada documento é identificado pelo `review_id`; o ID é enviado junto ao texto no contexto do
  gerador e exibido como evidência na interface.
- O pipeline se abstém quando não há documentos recuperados. Também pode receber
  `RETRIEVAL_MAX_DISTANCE` para remover evidências acima de um limite de distância cosine antes da
  geração. O valor exploratório `0.408` foi medido no índice completo; não é ativado por padrão e
  não deve ser usado com a amostra de desenvolvimento.
- A distância é um sinal aproximado de relevância, não um julgamento semântico. A calibração requer
  perguntas rotuladas e validação independente; veja [benchmark.md](benchmark.md).
- A busca e a calibração por distância foram exercitadas no índice completo de 38.624 reviews. A
  validação ponta a ponta com Gemini documentada cobre uma pergunta no índice de amostra de 2.000;
  ainda falta avaliar a geração com o índice completo e perguntas independentes.
- A amostra serve para desenvolvimento e demonstração, não para estimar quantitativamente a
  distribuição de opiniões de toda a base.

## Segurança e reprodutibilidade

- `GEMINI_API_KEY` é lida do ambiente local; nunca deve ser registrada em código ou logs.
- Dados brutos, vetores, cache do modelo e índice Chroma são locais e ignorados pelo Git.
- O guia de instalação em [getting-started.md](getting-started.md) descreve como recriar o ambiente,
  os dados e o índice.
