
# FIAP Data Science com IA: Tech Challenge Fase 4

Solucao de Voice of Customer com RAG sobre avaliacoes do Brazilian E-Commerce Public Dataset by Olist.

## Instalação e execução

Os comandos abaixo partem de um clone limpo e geram uma amostra local para desenvolvimento. Para
executar a base completa, veja a opção ao final da etapa 5.

### Pré-requisitos

- Python 3.11 ou 3.12
- Git

### 1. Clonar o repositório e criar/ativar o ambiente virtual

```bash
git clone https://github.com/Joaodanieljr/fiap-ds-ia-tech-challenge-4.git
cd fiap-ds-ia-tech-challenge-4
python -m venv .venv
```

No Windows, usando PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

Confirme que o ambiente está ativo: o prompt do terminal deve começar com `(.venv)`. Se o PowerShell
bloquear a ativação por política de execução, use `.\.venv\Scripts\python.exe` no lugar de `python`
nos comandos seguintes.

### 2. Instalar dependências da aplicação e da preparação de dados

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -r requirements.txt
```

Isso instala a interface Streamlit, o cliente Gemini, ChromaDB, o modelo local de embeddings,
as ferramentas de preparação/download dos dados e o pytest.

### 3. Configurar a chave Gemini

Copie o modelo de configuração:

```powershell
Copy-Item .env.example .env
```

Edite `.env` e preencha `GEMINI_API_KEY` com uma chave da API Gemini. O modelo padrão já está
configurado em `GEMINI_MODEL`. Nunca compartilhe ou versione `.env`; ele está ignorado pelo Git.
Não é necessário informar a chave para baixar os dados, gerar embeddings ou executar os testes
unitários, mas ela é necessária para gerar respostas reais com o Gemini.

### 4. Baixar os dados

```powershell
python src/baixar_dados.py
```

O script baixa o [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
via `kagglehub` e copia para `data/raw/` apenas estas tabelas:

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

### 5. Gerar o índice vetorial

```bash
python src/indexar_base.py --amostra 2000
```

O comando acima prepara uma amostra reprodutível de 2.000 avaliações e grava o índice Chroma em
`data/chroma/`. Os dados brutos baixados ficam em `data/raw/`. Na primeira execução, o modelo de
embeddings é baixado automaticamente; o tempo depende da conexão e do computador.

> **Atenção:** cada execução recria a coleção `avaliacoes_olist`. Gerar a amostra depois do índice
> completo (ou vice-versa) substitui a coleção anterior. Para indexar a base completa, rode
> `python src/indexar_base.py` no lugar do comando com `--amostra`.

### 6. Executar os testes automatizados

```powershell
python -m pytest -q
```

Os testes não chamam a API Gemini nem precisam de uma chave válida.

### 7. Fazer o teste ponta a ponta

Inicie a interface:

```powershell
streamlit run app.py
```

Se o Streamlit perguntar se deseja informar um e-mail para receber novidades, pressione Enter sem
preencher. Abra o endereço local mostrado no terminal (normalmente `http://localhost:8501`), faça uma
pergunta como **“Quais problemas aparecem nos relatos sobre produtos que chegaram quebrados?”** e
clique em **Buscar resposta**. Confira se a resposta é acompanhada pelos reviews citados e se os
`review_id`, textos e metadados exibidos são coerentes com a pergunta.

Para testar um caso fora do escopo, experimente **“Qual é a capital da França?”** e avalie a
resposta com atenção. **Limitação atual:** o pipeline se abstém quando a busca retorna zero
avaliações, mas ainda não aplica um limiar de distância calibrado para descartar resultados
irrelevantes. Portanto, uma pergunta fora do escopo pode recuperar reviews e não gerar abstinência;
não considere esse comportamento validado até a calibração ser implementada.

Se a API retornar `503 UNAVAILABLE`, o modelo pode estar temporariamente sobrecarregado; tente
novamente mais tarde. Erros de autenticação indicam que a chave ou as permissões da API precisam ser
conferidas no `.env`.

## Base de conhecimento

A base de conhecimento transforma as avaliações escritas pelos clientes em documentos pesquisáveis
semanticamente. As decisões abaixo foram tomadas a partir da análise exploratória em
[`notebooks/01_exploracao_base.ipynb`](notebooks/01_exploracao_base.ipynb) e estão implementadas em
[`src/preparar_base.py`](src/preparar_base.py) e [`src/indexar_base.py`](src/indexar_base.py).

**Resultado:** de 99.224 avaliações, **38.624 documentos** compõem a base final.

### 1. Seleção dos dados

| Etapa | Avaliações |
|---|---|
| Avaliações na base original | 99.224 |
| Com algum texto (título ou comentário) | 42.706 |
| Após remover `review_id` duplicados e textos com menos de 10 caracteres | **38.624** |

- **Só avaliações com texto entram na base.** O objetivo é explorar a voz do cliente, e avaliações
  apenas com nota não têm conteúdo para ser recuperado nem citado como evidência.
- **Apenas 4 das 9 tabelas do dataset são usadas** (avaliações, pedidos, itens e produtos): são as
  que trazem o texto e os metadados úteis para a recuperação. As demais (pagamentos, vendedores,
  geolocalização) não respondem às perguntas de negócio do desafio.

### 2. Limpeza e preparação dos textos

- **Documento = título + comentário.** O título falta em 88% das avaliações, mas, quando existe,
  costuma resumir o problema ("Produto com defeito") e reforça o sinal semântico.
- **Normalização mínima:** remoção de espaços extras e quebras de linha. **Não** removemos acentos,
  stopwords nem convertemos para minúsculas: modelos de embedding são treinados com texto natural,
  e a limpeza agressiva típica de NLP clássico (bag-of-words) prejudica a busca semântica.
- **Descarte de textos com menos de 10 caracteres** (3.199 casos). São quase todos elogios genéricos
  ("Bom", "Ótimo", "Ok", "."), que não dizem *o que* agradou ao cliente e competiriam na busca com
  comentários que citam aspectos concretos (entrega, qualidade, atendimento).

### 3. Tratamento de valores ausentes

| Campo | Situação | Tratamento |
|---|---|---|
| Título e comentário | 88% e 59% nulos | Concatenação do que existir; sem texto algum, a avaliação sai da base |
| Categoria do produto | 1.389 pedidos sem categoria | Preenchida com `sem_categoria` |
| Data de entrega | 2.965 pedidos sem data (cancelados, em trânsito etc.) | Classificados como `nao_entregue` |

O ChromaDB não aceita metadados nulos, e um `assert` em `preparar_base()` garante que nenhum
documento seja indexado com campo vazio.

### 4. Organização dos documentos

- **1 avaliação = 1 documento, sem chunking.** O maior comentário tem 208 caracteres, bem abaixo do
  limite de entrada do modelo de embedding. Dividir textos tão curtos só fragmentaria o contexto.
- **ID do documento = `review_id`.** Foram removidos 814 IDs repetidos (o mesmo review replicado em
  pedidos comprados juntos), garantindo um identificador único e rastreável: é por ele que a
  resposta cita as evidências usadas.
- **Metadados por documento:**

| Metadado | Tipo | Origem | Uso na recuperação |
|---|---|---|---|
| `nota` | int (1–5) | avaliação | Filtrar clientes satisfeitos ou insatisfeitos |
| `categoria` | str | produto | Restringir a busca a uma categoria |
| `data_compra` | str (AAAA-MM-DD) | pedido | Exibição nas evidências |
| `ano_mes` | int (ex.: 201803) | pedido | Filtrar por período (permite `$gte`/`$lte`) |
| `situacao_entrega` | `no_prazo` / `atrasado` / `nao_entregue` | pedido | Investigar experiências de entrega |

- **Pedidos com mais de uma categoria** (727, cerca de 0,7%): usamos a categoria do item de maior
  preço, considerado o produto principal. Isso também evita que o join com os itens duplique a
  avaliação na base.
- **Atraso:** comparação, em dias, entre a data de entrega ao cliente e a data prevista. Resultado:
  33.105 entregues no prazo, 3.695 atrasados e 1.824 não entregues.

**Como os metadados melhoram a recuperação:** a busca semântica sozinha encontra textos *parecidos*
com a pergunta, mas não sabe se o cliente estava satisfeito ou se a entrega atrasou. Perguntas como
"quais problemas os clientes insatisfeitos relatam?" ou "o que reclamam sobre entregas atrasadas?"
ficam mais precisas quando a busca é restrita por `nota` ou `situacao_entrega`, porque o contexto
enviado ao modelo deixa de misturar elogios e reclamações.

### 5. Criação dos embeddings

- **Modelo:** [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
  (sentence-transformers), executado localmente.
- **Por que este modelo:**
  - **Multilíngue:** as avaliações estão em português; um modelo treinado só em inglês recuperaria mal.
  - **Leve e rápido em CPU:** vetores de 384 dimensões; a base completa é indexada em poucos minutos,
    sem GPU.
  - **Sem prefixos de consulta:** modelos como a família e5 exigem os prefixos `query:`/`passage:`.
    Esquecê-los degrada a busca sem gerar erro, um risco desnecessário num pipeline mantido por
    várias pessoas.
  - **Local e gratuito:** não consome cota de API nem depende de chave para gerar a base, o que
    facilita a reprodução da solução.
- **Vetores normalizados** (`normalize_embeddings=True`), o que torna a similaridade de cosseno direta
  e estável. A pergunta do usuário **precisa** ser vetorizada com o mesmo modelo e a mesma
  normalização.

### 6. Armazenamento e indexação

- **ChromaDB em modo persistente local** (`data/chroma/`): banco vetorial embutido, que não exige
  servidor nem conta em nuvem e suporta filtros por metadado na mesma consulta da busca semântica.
- **Coleção `avaliacoes_olist` com distância de cosseno** (`hnsw:space = cosine`). O padrão do Chroma
  é a distância euclidiana; com cosseno, as distâncias ficam numa escala fixa, o que torna o limiar
  de "evidência insuficiente" interpretável.
- **Índice HNSW**, a estrutura de busca aproximada de vizinhos mais próximos do Chroma, que responde
  em milissegundos mesmo com dezenas de milhares de documentos.
- **Reprodutibilidade:** o índice não é versionado no Git; ele é gerado do zero por
  `src/indexar_base.py`, que recria a coleção a cada execução (rodar duas vezes não duplica
  documentos). A opção `--amostra` usa `random_state=42`, então todos obtêm a mesma amostra.

### Validação da recuperação

Teste no índice completo ([`notebooks/02_teste_busca.ipynb`](notebooks/02_teste_busca.ipynb)), com
distância de cosseno (quanto menor, mais parecido):

| Pergunta | Distâncias (top 5) | Resultado |
|---|---|---|
| "produto chegou quebrado ou com defeito" (filtro `nota <= 2`) | 0,12 – 0,13 | Todos os textos relatam produto quebrado ou com defeito, em 5 categorias diferentes |
| "qual é a capital da França?" (fora do escopo) | 0,52 – 0,65 | Textos genéricos, sem relação com a pergunta |

A separação observada nessas consultas é uma referência exploratória, não um limiar já aplicado
pelo pipeline. Na amostra de 2 mil documentos as distâncias podem variar; qualquer limiar de
abstinência precisa ser avaliado com exemplos rotulados e na mesma configuração de índice que será
usada na execução final.
