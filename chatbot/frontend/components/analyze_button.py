import streamlit as st

from chatbot.frontend.components.markdowns import style_seg_button
from chatbot.backend.services.analyze import analyze_essay


def click_analyze_button():
    if st.session_state.get("analyzed_text", None) is None:
        with st.container(key="seg_spinner"):
            with st.spinner("Segmenting...", show_time=True):
                analyzed_text, predictions = analyze_essay(st.session_state.essay_text)
                st.session_state.analyzed_text = analyzed_text
                st.session_state.analyzed_predictions = " ".join(predictions)

    st.session_state.chat_history.append(
        {
            "role": "analyzer",
            "content": st.session_state.analyzed_text,
            "predictions": st.session_state.analyzed_predictions,
        }
    )


def get_analyze_button():
    _, _, col_right = st.columns([1, 2, 1])
    with col_right:
        st.button(
            "Segmentation",
            on_click=click_analyze_button,
            icon=":material/library_books:",
            key="seg_button",
        )
        style_seg_button()
