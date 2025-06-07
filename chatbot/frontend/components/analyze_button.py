import streamlit as st

from components.markdowns import style_seg_button
from backend.services.analyze import analyze_essay


def click_analyze_button():
    if st.session_state.get("analyzed_text", None) is None:
        with st.chat_message("assistant"):
            with st.spinner("Segmenting..."):
                analyzed_text = analyze_essay(st.session_state.essay_text)
                st.session_state.analyzed_text = analyzed_text

    st.session_state.chat_history.append(
        {"role": "analyzer", "content": st.session_state.analyzed_text}
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
