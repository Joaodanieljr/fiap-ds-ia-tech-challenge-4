"""Mede distâncias da bateria no índice escolhido, sem chamar o Gemini."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from voice_of_customer.config import load_settings
from voice_of_customer.retrieval import ChromaReviewRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mede a distância dos resultados da bateria sem usar a API Gemini."
    )
    parser.add_argument(
        "--persist-directory",
        default="./data/chroma",
        help="diretório do índice Chroma a avaliar",
    )
    parser.add_argument(
        "--collection-name",
        default="avaliacoes_olist",
        help="coleção Chroma a avaliar",
    )
    parser.add_argument(
        "--output",
        default="docs/abstention_calibration.json",
        help="arquivo JSON de saída, relativo à raiz do projeto",
    )
    args = parser.parse_args()

    settings = replace(
        load_settings(),
        chroma_persist_directory=args.persist_directory,
        chroma_collection_name=args.collection_name,
        retrieval_max_distance=None,
    )
    retriever = ChromaReviewRetriever(settings)
    benchmark_path = PROJECT_ROOT / "docs" / "resultado_benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    results = []

    for item in benchmark:
        evidence = retriever.retrieve(
            question=item["pergunta"],
            limit=settings.retrieval_top_k,
            where=item.get("filtro_aplicado"),
        )
        distances = [
            evidence_item.distance
            for evidence_item in evidence
            if evidence_item.distance is not None
        ]
        results.append(
            {
                "id": item["id"],
                "categoria": item["categoria"],
                "pergunta": item["pergunta"],
                "filtro_aplicado": item.get("filtro_aplicado"),
                "qtd_evidencias": len(evidence),
                "evidencias_review_ids": [item.review_id for item in evidence],
                "distancias": distances,
                "menor_distancia": min(distances) if distances else None,
            }
        )

    in_scope = [
        item["menor_distancia"]
        for item in results
        if item["categoria"] != "Fora do Escopo"
        and item["menor_distancia"] is not None
    ]
    out_of_scope = [
        item["menor_distancia"]
        for item in results
        if item["categoria"] == "Fora do Escopo"
        and item["menor_distancia"] is not None
    ]
    suggested_threshold = None
    separation = None
    conservative_threshold = None
    conservative_result = None
    if in_scope and out_of_scope:
        last_in_scope = max(in_scope)
        first_out_of_scope = min(out_of_scope)
        separation = last_in_scope < first_out_of_scope
        if separation:
            suggested_threshold = (last_in_scope + first_out_of_scope) / 2
        conservative_threshold = math.floor(first_out_of_scope * 1000) / 1000
        accepted_in_scope = [
            item["id"]
            for item in results
            if item["categoria"] != "Fora do Escopo"
            and item["menor_distancia"] is not None
            and item["menor_distancia"] <= conservative_threshold
        ]
        accepted_out_of_scope = [
            item["id"]
            for item in results
            if item["categoria"] == "Fora do Escopo"
            and item["menor_distancia"] is not None
            and item["menor_distancia"] <= conservative_threshold
        ]
        conservative_result = {
            "perguntas_em_escopo_respondidas": accepted_in_scope,
            "perguntas_em_escopo_abstidas": [
                item["id"]
                for item in results
                if item["categoria"] != "Fora do Escopo"
                and item["menor_distancia"] is not None
                and item["menor_distancia"] > conservative_threshold
            ],
            "perguntas_fora_escopo_respondidas": accepted_out_of_scope,
            "perguntas_fora_escopo_abstidas": [
                item["id"]
                for item in results
                if item["categoria"] == "Fora do Escopo"
                and (
                    item["menor_distancia"] is None
                    or item["menor_distancia"] > conservative_threshold
                )
            ],
        }

    output = {
        "executado_em_utc": datetime.now(timezone.utc).isoformat(),
        "indice": {
            "diretorio": args.persist_directory,
            "colecao": args.collection_name,
            "quantidade_documentos": retriever.collection.count(),
            "top_k": settings.retrieval_top_k,
            "modelo_embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "distancia": "cosine",
        },
        "criterio_exploratorio": {
            "perguntas_fora_escopo_separaveis_por_distancia": separation,
            "maior_menor_distancia_em_escopo": max(in_scope) if in_scope else None,
            "menor_menor_distancia_fora_escopo": (
                min(out_of_scope) if out_of_scope else None
            ),
            "limite_candidato": suggested_threshold,
            "limite_conservador_sem_falso_aceite_nesta_bateria": (
                conservative_threshold
            ),
            "resultado_limite_conservador": conservative_result,
            "observacao": (
                "O limite é apenas um candidato exploratório derivado destas 15 perguntas; "
                "precisa de validação independente antes de ser tratado como garantia. "
                "O limite conservador prioriza rejeitar perguntas fora do escopo e pode "
                "abster perguntas válidas."
            ),
        },
        "resultados": results,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Relatório salvo em {output_path}")
    print(
        "Separação perfeita: "
        f"{separation}; limite perfeito: {suggested_threshold}; "
        f"limite conservador: {conservative_threshold}"
    )


if __name__ == "__main__":
    main()
