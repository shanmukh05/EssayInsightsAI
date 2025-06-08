import streamlit as st
import os
import torch
import sys

torch.classes.__path__ = []
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from chatbot.frontend.components.markdowns import (
    get_css_style,
    get_intro_html,
    get_details_html,
)
from chatbot.frontend.components.side_panel import get_side_panel
from chatbot.frontend.components.input_box import get_input_section
from chatbot.frontend.components.analyze_button import get_analyze_button
from chatbot.backend.services.assistant import generate_chat_reply

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Essay Assistant",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Chat Message Alignment ---
get_css_style()

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "essay_text" not in st.session_state:
    st.session_state.essay_text = ""
if "chat_enabled" not in st.session_state:
    st.session_state.chat_enabled = False

# --- Sidepanel ---
openai_api_key, selected_model = get_side_panel()

# --- Main Content Area ---
if not st.session_state.chat_enabled:
    # --- Introduction ---
    get_intro_html()

    # --- User Input Section ---
    get_input_section()

    # --- More Details ---
    get_details_html()


# --- Chat Interface ---
if st.session_state.chat_enabled:
    st.markdown(
        "<h1 style='position:fixed; top:30px; left:400px;'> 🤖 EssayBot</h1>",
        unsafe_allow_html=True,
    )

    # Display chat messages from history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(message["content"], unsafe_allow_html=True)
        elif message["role"] == "analyzer":
            with st.chat_message("assistant"):
                st.write(message["content"], unsafe_allow_html=True)

    get_analyze_button()

    # Chat input at the bottom
    user_query = st.chat_input("Ask me about your essay or ask for feedback...")

    if user_query:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_query)

        # Simulate bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                bot_response = generate_chat_reply(
                    user_query=user_query,
                    essay_text=st.session_state.essay_text,
                    essay_analysis=st.session_state.get(
                        "analyzed_predictions", "Analysis Not Present"
                    ),
                    chat_history=st.session_state.chat_history,
                    model=selected_model,
                    openai_key=openai_api_key,
                )

                st.write(bot_response)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": bot_response}
                )
