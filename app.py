"""Interface Streamlit inicial para perguntas sobre reviews Olist."""

import streamlit as st

from voice_of_customer.config import load_settings
from voice_of_customer.generation import GeminiAnswerGenerator
from voice_of_customer.pipeline import RagPipeline
from voice_of_customer.retrieval import ChromaReviewRetriever


st.set_page_config(page_title="Olist Voice of Customer", page_icon="🛍️")
st.title("Voice of Customer | Olist")
st.write("Explore as experiencias relatadas pelos clientes nas avaliacoes.")

question = st.text_input("Pergunta", placeholder="Ex.: Quais problemas aparecem nos relatos sobre entrega?")

if st.button("Buscar resposta", type="primary", disabled=not question.strip()):
    try:
        settings = load_settings()
        pipeline = RagPipeline(
            retriever=ChromaReviewRetriever(settings),
            generator=GeminiAnswerGenerator(settings),
            top_k=settings.retrieval_top_k,
            max_distance=settings.retrieval_max_distance,
        )
        with st.spinner("Buscando evidencias e preparando resposta..."):
            result = pipeline.answer(question.strip())
    except Exception as error:
        st.error(f"Nao foi possivel executar o pipeline: {error}")
    else:
        st.markdown(result.text)
        if result.abstained:
            st.info("Resposta abstida: nao foram encontradas evidencias suficientes.")
        for item in result.evidence:
            with st.expander(f"Review {item.review_id}"):
                st.write(item.text)
                if item.distance is not None:
                    st.caption(f"Distancia retornada pelo Chroma: {item.distance:.4f}")
                if item.metadata:
                    st.json(item.metadata)
