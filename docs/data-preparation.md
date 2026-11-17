
# Preparação da base de conhecimento

A base de conhecimento transforma as avaliações escritas pelos clientes em documentos pesquisáveis
semanticamente. As decisões abaixo foram tomadas a partir da análise exploratória em
[`notebooks/01_exploracao_base.ipynb`](../notebooks/01_exploracao_base.ipynb) e estão implementadas em
[`src/preparar_base.py`](../src/preparar_base.py) e [`src/indexar_base.py`](../src/indexar_base.py).

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

Teste no índice completo ([`notebooks/02_teste_busca.ipynb`](../notebooks/02_teste_busca.ipynb)), com
distância de cosseno (quanto menor, mais parecido):

| Pergunta | Distâncias (top 5) | Resultado |
|---|---|---|
| "produto chegou quebrado ou com defeito" (filtro `nota <= 2`) | 0,12 – 0,13 | Todos os textos relatam produto quebrado ou com defeito, em 5 categorias diferentes |
| "qual é a capital da França?" (fora do escopo) | 0,52 – 0,65 | Textos genéricos, sem relação com a pergunta |

A separação observada nessas consultas é uma referência exploratória, não um limiar já aplicado
pelo pipeline. Na amostra de 2 mil documentos as distâncias podem variar; qualquer limiar de
abstinência precisa ser avaliado com exemplos rotulados e na mesma configuração de índice que será
usada na execução final.
