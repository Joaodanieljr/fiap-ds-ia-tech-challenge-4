# Guia de execução local

Este guia parte de um clone limpo e cria um índice de amostra. Os comandos de shell são para
PowerShell no Windows, salvo indicação contrária.

## Requisitos

- Python 3.11 ou 3.12
- Git
- Acesso à internet para baixar os dados e o modelo de embeddings
- Chave da API Gemini para testar respostas geradas
- Conta/token Kaggle apenas se `kagglehub` solicitar autenticação

## 1. Clonar e criar o ambiente virtual

```powershell
git clone https://github.com/Joaodanieljr/fiap-ds-ia-tech-challenge-4.git
Set-Location fiap-ds-ia-tech-challenge-4
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Quando ativo, o terminal mostra `(.venv)`. Se a política do PowerShell bloquear a ativação, não é
necessário alterar a política: use `.\.venv\Scripts\python.exe` para executar o Python nos comandos
abaixo.

No Linux/macOS, ative com `source .venv/bin/activate`.

## 2. Instalar dependências

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O arquivo de requisitos instala o projeto e os extras de desenvolvimento, download/preparação de
dados e notebooks a partir das dependências declaradas em `pyproject.toml`.

## 3. Configurar a API Gemini

```powershell
Copy-Item .env.example .env
notepad .env
```

Preencha `GEMINI_API_KEY` com sua chave e salve. O modelo padrão já está indicado em `GEMINI_MODEL`.
Não envie a chave no chat e não versione o `.env`; o arquivo é ignorado pelo Git.

A chave é necessária para responder na interface, mas não para baixar os dados, construir o índice
ou executar os testes automatizados.

## 4. Baixar os dados Olist

```powershell
python src/baixar_dados.py
```

O script baixa o [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
e copia para `data/raw/` as tabelas necessárias. Caso seja solicitada autenticação, crie um token em
Kaggle → Settings → API e siga as instruções oficiais do Kaggle para armazená-lo localmente. Não
inclua o token no repositório.

## 5. Construir o índice

Para desenvolvimento, use uma amostra reprodutível:

```powershell
python src/indexar_base.py --amostra 2000
```

O índice persistido será criado em `data/chroma/`, na coleção `avaliacoes_olist`. Na primeira
execução, o modelo de embeddings é baixado. A duração depende da máquina e da conexão.

Para construir o índice completo, rode `python src/indexar_base.py` sem `--amostra`. **O script
recria a coleção toda vez:** criar uma amostra ou um índice completo substitui a coleção anterior.
Não execute esse comando se quiser preservar outro índice local.

## 6. Executar os testes automatizados

```powershell
python -m pytest -q
```

Os testes não chamam Chroma, Gemini ou Kaggle, e não exigem uma chave válida.

## 7. Executar a bateria exploratória (opcional)

Com o índice construído e `GEMINI_API_KEY` configurada, execute:

```powershell
python tests/test_bateria_benchmark.py
```

O script faz até 15 chamadas ao Gemini e grava as respostas em
`docs/resultado_benchmark.json`, substituindo o relatório existente. Isso não é necessário para
rodar os testes automatizados. As limitações e a interpretação dos resultados estão em
[benchmark.md](benchmark.md).

## 8. Fazer o teste ponta a ponta

```powershell
streamlit run app.py
```

Se o Streamlit perguntar se deseja informar um e-mail para receber novidades, pressione Enter sem
preencher. Abra o endereço local indicado no terminal, normalmente `http://localhost:8501`.

Experimente:

> Quais problemas aparecem nos relatos sobre produtos que chegaram quebrados?

Clique em **Buscar resposta**. Confira se a resposta apresenta os reviews usados e se cada `review_id`
mostrado na resposta está entre as evidências expandidas na tela.

### Erros comuns

- **403/401:** confira a chave e as permissões no `.env`, sem compartilhá-la.
- **503 UNAVAILABLE:** o serviço/modelo pode estar sob alta demanda; aguarde e tente novamente.
- **Coleção inexistente:** verifique se executou o passo 5 e se `CHROMA_PERSIST_DIRECTORY` e
  `CHROMA_COLLECTION_NAME` correspondem ao índice criado.
- **Download Kaggle falhou:** confira a conexão e se o `kagglehub` exige autenticação.

Uma pergunta fora do domínio, como “Qual é a capital da França?”, pode ser usada para observar os
limites do sistema. No estado atual, o pipeline se abstém se a busca retornar zero reviews, mas
ainda não tem limiar de distância calibrado para rejeitar resultados irrelevantes.
