import streamlit as st


def get_input_section():
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your essay file (e.g., .txt, .md)",
            type=["txt", "md"],
            help="Upload a text-based essay file.",
            key="file_uploader",
        )
        if uploaded_file is not None:
            # Simulate file conversion to text
            # In a real app:
            # if uploaded_file.type == "application/pdf":
            #     text = extract_text_from_pdf(uploaded_file)
            # elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            #     text = extract_text_from_docx(uploaded_file)
            # else:
            text = uploaded_file.read().decode("utf-8")
            st.session_state.essay_text = text
            st.session_state.chat_enabled = True
            st.session_state.chat_history.append(
                {"role": "user", "content": f"Here is my essay:\n\n{text}"}
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
                {"role": "user", "content": f"Here is my essay:\n\n{pasted_text}"}
            )
            st.rerun()
