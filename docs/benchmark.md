# Avaliação exploratória do pipeline RAG

A bateria contém 15 perguntas para exercitar a síntese de avaliações, os filtros de metadados e
perguntas fora do escopo. Ela é uma avaliação exploratória, não um teste de precisão/recall: não há
gabarito de documentos relevantes nem julgamento independente da qualidade de cada resposta.

## Cobertura

- **Enunciado (P01–P06):** reclamações, elogios, qualidade e entrega.
- **Metadados (P07–P11):** nota 1 e 5, categoria `perfumaria`, situação `atrasado` e período de 2017
  (`ano_mes` entre 201701 e 201712).
- **Fora do escopo (P12–P15):** passagens aéreas, informações corporativas, garantia de produto e
  instruções do aplicativo.

## Resultado registrado

O relatório entregue contém 15 respostas sem erro de execução, cada uma com cinco reviews
recuperados e seus `review_id`s. Isso confirma a execução do fluxo e dos filtros usados, mas **não
confirma que os cinco reviews sejam relevantes** nem mede a correção factual das respostas.

Nas quatro perguntas fora do escopo, o Chroma também retornou cinco reviews. O pipeline atual só se
abstém quando não há resultados; portanto, uma frase do Gemini dizendo que as evidências são
insuficientes não demonstra que o recuperador detectou a falta de evidência. Além disso, a resposta
P13 não cita os `review_id`s retornados. A bateria não deve ser apresentada como comprovação formal
de ausência de alucinações ou de abstinência calibrada.

O JSON entregue não registra a data da execução nem o tamanho/configuração do índice usado. Para
reproduzir a avaliação, reconstrua o índice conforme [o guia de execução](getting-started.md), confira
o `.env` e rode o comando abaixo. A execução faz chamadas reais ao Gemini e substitui o relatório
registrado.

```powershell
python tests/test_bateria_benchmark.py
```

## Artefatos

- Bateria executável e testes de consistência: [tests/test_bateria_benchmark.py](../tests/test_bateria_benchmark.py)
- Respostas e evidências registradas: [docs/resultado_benchmark.json](resultado_benchmark.json)
- Decisões de preparação da base: [docs/data-preparation.md](data-preparation.md)