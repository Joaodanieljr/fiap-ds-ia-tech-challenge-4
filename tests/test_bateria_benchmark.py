import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "src"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from voice_of_customer.config import load_settings
from voice_of_customer.generation import GeminiAnswerGenerator
from voice_of_customer.pipeline import RagPipeline
from voice_of_customer.retrieval import ChromaReviewRetriever

BATERIA_TESTES = [
    {"id": "P01", "cat": "Enunciado", "pergunta": "Quais são os principais motivos de insatisfação e problemas relatados pelos clientes?", "filtro": None},
    {"id": "P02", "cat": "Enunciado", "pergunta": "Quais reclamações e problemas aparecem com maior frequência nas avaliações relacionadas à entrega?", "filtro": None},
    {"id": "P03", "cat": "Enunciado", "pergunta": "O que os clientes mais elogiam em suas experiências de compra?", "filtro": None},
    {"id": "P04", "cat": "Enunciado", "pergunta": "Quais problemas e padrões podem ser identificados nas avaliações de clientes insatisfeitos?", "filtro": None},
    {"id": "P05", "cat": "Enunciado", "pergunta": "Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?", "filtro": None},
    {"id": "P06", "cat": "Enunciado", "pergunta": "Quais evidências sustentam a conclusão de que existem atrasos na entrega dos pedidos?", "filtro": None},
    {"id": "P07", "cat": "Metadados", "pergunta": "Quais são as principais reclamações nas avaliações com nota 1?", "filtro": {"nota": 1}},
    {"id": "P08", "cat": "Metadados", "pergunta": "O que os clientes mais elogiam nas avaliações que deram nota 5?", "filtro": {"nota": 5}},
    {"id": "P09", "cat": "Metadados", "pergunta": "Quais são os principais relatos de insatisfação na categoria perfumaria?", "filtro": {"categoria": "perfumaria"}},
    {"id": "P10", "cat": "Metadados", "pergunta": "Quais problemas de entrega foram relatados nas compras do ano de 2017?", "filtro": {"$and": [{"ano_mes": {"$gte": 201701}}, {"ano_mes": {"$lte": 201712}}]}},
    {"id": "P11", "cat": "Metadados", "pergunta": "O que os clientes reclamam em pedidos com entrega atrasada?", "filtro": {"situacao_entrega": "atrasado"}},
    {"id": "P12", "cat": "Fora do Escopo", "pergunta": "Qual é a política de reembolso para voos internacionais e passagens aéreas na Olist?", "filtro": None},
    {"id": "P13", "cat": "Fora do Escopo", "pergunta": "Quem é o atual CEO da Olist e qual o faturamento da empresa em 2025?", "filtro": None},
    {"id": "P14", "cat": "Fora do Escopo", "pergunta": "Qual é a garantia para aspiradores de pó Electrolux comprados com cupom?", "filtro": None},
    {"id": "P15", "cat": "Fora do Escopo", "pergunta": "Como faço para atualizar meu endereço de entrega pelo aplicativo da Olist?", "filtro": None},
]

def rodar_benchmark() -> int:
    resultados = []
    print("Carregando configuracoes e inicializando pipeline...")

    settings = load_settings()
    retriever = ChromaReviewRetriever(settings=settings)
    generator = GeminiAnswerGenerator(settings=settings)

    pipeline = RagPipeline(
        retriever=retriever,
        generator=generator,
        top_k=settings.retrieval_top_k,
        max_distance=settings.retrieval_max_distance,
    )

    print("Executando Bateria de Testes")

    for item in BATERIA_TESTES:
        print(f"Testando [{item['id']}] - {item['pergunta']}...")

        resposta_obj = None
        erro_msg = None

        for tentativa in range(3):
            try:
                resposta_obj = pipeline.answer(question=item["pergunta"], where=item["filtro"])
                break
            except Exception as erro:
                erro_msg = str(erro)
                if "429" in erro_msg or "RESOURCE_EXHAUSTED" in erro_msg:
                    if tentativa == 2:
                        break
                    print("Limite de requisicoes atingido (429). Aguardando 15s...")
                    time.sleep(15)
                elif "503" in erro_msg or "UNAVAILABLE" in erro_msg:
                    if tentativa == 2:
                        break
                    print("Instabilidade na API. Aguardando 4s...")
                    time.sleep(4)
                else:
                    break

        resultado = {
            "id": item["id"],
            "categoria": item["cat"],
            "pergunta": item["pergunta"],
            "filtro_aplicado": item["filtro"],
        }
        if resposta_obj is not None:
            texto_resposta = getattr(resposta_obj, "text", str(resposta_obj))
            evidencias = getattr(resposta_obj, "evidence", [])
            evidencias_ids = [ev.review_id for ev in evidencias if hasattr(ev, 'review_id')]
            resultado.update(
                {
                    "resposta_gerada": texto_resposta,
                    "qtd_evidencias": len(evidencias_ids),
                    "evidencias_review_ids": evidencias_ids,
                    "abstained": resposta_obj.abstained,
                    "evidencias_distancias": [
                        ev.distance for ev in evidencias if hasattr(ev, "distance")
                    ],
                }
            )
        else:
            resultado["erro"] = erro_msg
        resultados.append(resultado)

    pasta_docs = PROJECT_ROOT / "docs"
    pasta_docs.mkdir(exist_ok=True)
    out_path = pasta_docs / "resultado_benchmark.json"
    out_path.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    erros = sum("erro" in resultado for resultado in resultados)
    print(
        f"\nBenchmark encerrado: {len(resultados)} cenarios, {erros} erros. "
        f"Relatorio salvo em {out_path}"
    )
    return erros

if __name__ == "__main__":
    raise SystemExit(rodar_benchmark())


def test_bateria_cobre_os_cenarios_planejados() -> None:
    assert len(BATERIA_TESTES) == 15
    assert [item["id"] for item in BATERIA_TESTES] == [
        f"P{numero:02d}" for numero in range(1, 16)
    ]
    assert sum(item["cat"] == "Enunciado" for item in BATERIA_TESTES) == 6
    assert sum(item["cat"] == "Metadados" for item in BATERIA_TESTES) == 5
    assert sum(item["cat"] == "Fora do Escopo" for item in BATERIA_TESTES) == 4


def test_relatorio_entregue_tem_resultados_consistentes() -> None:
    report_path = PROJECT_ROOT / "docs" / "resultado_benchmark.json"
    resultados = json.loads(report_path.read_text(encoding="utf-8"))

    assert [resultado["id"] for resultado in resultados] == [
        item["id"] for item in BATERIA_TESTES
    ]
    for resultado in resultados:
        assert "erro" not in resultado
        assert "resposta_gerada" in resultado
        assert resultado["qtd_evidencias"] == len(
            resultado["evidencias_review_ids"]
        )
        if resultado.get("abstained"):
            assert resultado["qtd_evidencias"] == 0
        else:
            assert resultado["qtd_evidencias"] > 0