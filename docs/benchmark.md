# Avaliação exploratória do pipeline RAG

A bateria contém 15 perguntas para exercitar a síntese de avaliações, os filtros de metadados e
perguntas fora do escopo. Ela é uma avaliação exploratória, não um teste de precisão/recall: não há
gabarito de documentos relevantes nem julgamento independente da qualidade de cada resposta.

## Calibrar a abstinência por distância

O pipeline recebe `RETRIEVAL_MAX_DISTANCE`, um limite inclusivo para a distância cosine: evidências
acima do limite (ou sem distância) são removidas antes de chamar o Gemini. Se nenhuma evidência
passar, o pipeline se abstém. A configuração de entrega usa `0.408`; para a amostra de
desenvolvimento, deixe o valor vazio.

Para medir as mesmas 15 perguntas em um índice completo sem substituir a amostra local, construa uma
coleção em outro diretório e execute a calibração, sem chamadas ao Gemini:

```powershell
python src/indexar_base.py --persist-directory .\data\chroma_full_calibration
python scripts/calibrar_abstencao.py --persist-directory .\data\chroma_full_calibration
```

O relatório `docs/abstention_calibration.json` inclui as distâncias, IDs e o número de documentos do
índice. O limite candidato só é sugerido quando, neste conjunto, a maior melhor-distância em escopo
fica abaixo da menor melhor-distância fora de escopo. O script também mostra o resultado de um limite
conservador que rejeita as perguntas fora de escopo observadas, junto com as perguntas válidas que
também seriam abstidas.

Na calibração atual, com **38.624 reviews**, não há um limite que separe perfeitamente as 11
perguntas em escopo das quatro fora de escopo: P07 (nota 1) tem melhor distância `0.4268`, enquanto
P14 (garantia de aspirador) tem `0.4086`. O limite conservador `0.408` rejeita as quatro perguntas
fora de escopo, mas também causa abstinência em P07. No teste do pipeline, P09 passa com duas
evidências (as demais perguntas aceitas passam com cinco). Portanto, a distância é um sinal útil,
mas não resolve sozinha todas as perguntas; configurar esse limite é uma escolha entre segurança e
cobertura. As medições usam apenas 15 perguntas e exigem validação independente antes da entrega.

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

Nas quatro perguntas fora do escopo, o Chroma retorna reviews, mas com o limite configurado o
pipeline se abstém antes de chamar o Gemini. A resposta P13 do relatório original não cita os
`review_id`s retornados. A bateria não deve ser apresentada como comprovação formal de ausência de
alucinações ou de abstinência calibrada para consultas novas.

O JSON entregue não registra a data da execução nem o tamanho/configuração do índice usado. Para
reproduzir as respostas, reconstrua o índice conforme [o guia de execução](getting-started.md),
confira o `.env` e rode o comando abaixo. A execução faz chamadas reais ao Gemini e substitui o
relatório registrado.

```powershell
python tests/test_bateria_benchmark.py
```

## Artefatos

- Bateria executável e testes de consistência: [tests/test_bateria_benchmark.py](../tests/test_bateria_benchmark.py)
- Respostas e evidências registradas: [docs/resultado_benchmark.json](resultado_benchmark.json)
- Decisões de preparação da base: [docs/data-preparation.md](data-preparation.md)