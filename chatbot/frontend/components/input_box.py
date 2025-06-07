import streamlit as st

from backend.services.extractor import extract_text_from_file


def get_input_section():
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your essay file (e.g., .txt, .jpeg, .jpg, .png, .pdf)",
            type=["txt", "pdf", "jpeg", "jpg", "png"],
            help="Upload a essay file.",
            key="file_uploader",
        )
        if uploaded_file is not None:
            with st.spinner("Extracting text...", show_time=True):
                text = extract_text_from_file(uploaded_file)
            st.session_state.essay_text = text
            st.session_state.chat_enabled = True
            st.session_state.chat_history.append(
                {"role": "user", "content": f"Here is the essay:\n\n{text}"}
            )
            st.rerun()

    with col2:
        pasted_text = st.text_area(
            "Or paste your essay text here",
            height=133,
            placeholder="Paste your essay content here...",
            key="text_area_input",
        )
        if pasted_text:
            st.session_state.essay_text = pasted_text
            st.session_state.chat_enabled = True
            st.session_state.chat_history.append(
                {"role": "user", "content": f"Here is the essay:\n\n{pasted_text}"}
            )
            st.rerun()
