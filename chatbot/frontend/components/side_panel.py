import streamlit as st


def get_sidepanel():
    with st.sidebar:
        st.image(
            r"D:\Learning\NLP\Projects\EssayInsightsAI\chatbot\frontend\assets\logo.png",
            use_container_width=True,
        )

        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            if st.button("➕ New Chat", key="new_chat_button"):
                st.session_state.chat_history = []
                st.session_state.essay_text = ""
                st.session_state.chat_enabled = False
                st.session_state.analyzed_text = None
                st.rerun()

        st.title("Settings")

        # OpenAI API Key Input
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key here. This will not be saved.",
            key="openai_api_key_input",
        )
        if openai_api_key:
            # In a real app, you might want to validate the key or store it securely
            st.success("API Key set!")
        else:
            st.warning("Please enter your OpenAI API key.")

        # Model Selection Dropdown
        model_options = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=0,  # Default to gpt-4o
            help="Choose the OpenAI model for generating responses.",
            key="model_selection",
        )
        st.info(f"Selected Model: {selected_model}")
