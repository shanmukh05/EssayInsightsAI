import streamlit as st
import os
import torch

torch.classes.__path__ = []
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from components.markdowns import get_css_style, get_intro_html, get_details_html
from components.sidepanel import get_sidepanel
from components.input_box import get_input_section
from chatbot.frontend.components.analyze_button import get_analyze_button

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Essay Assistant",
    page_icon="✍️",
    layout="centered",
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
get_sidepanel()

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
        "<h1 style='position:fixed; top:30px; left:350px;'> 🤖 EssayBot</h1>",
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

        # Display user message immediately (will be right-aligned by CSS)
        with st.chat_message("user"):
            st.write(user_query)

        # Simulate bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # import openai
                # openai.api_key = openai_api_key
                # try:
                #     response = openai.chat.completions.create(
                #         model=selected_model,
                #         messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history],
                #         max_tokens=500
                #     )
                #     bot_response = response.choices[0].message.content
                # except Exception as e:
                #     bot_response = f"Error communicating with OpenAI: {e}"

                # For demonstration, a simple mock response:
                if "feedback" in user_query.lower():
                    bot_response = "I can provide feedback on your essay's structure, clarity, and arguments. What specific aspect would you like me to focus on?"
                elif "summary" in user_query.lower():
                    bot_response = "To summarize your essay, I would highlight the main thesis and key supporting points. Would you like me to proceed with that?"
                else:
                    bot_response = "Thank you for your question! I'm ready to help you with your essay. What would you like to discuss next?"

                st.write(bot_response)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": bot_response}
                )
