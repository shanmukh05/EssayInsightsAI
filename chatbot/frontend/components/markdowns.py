import streamlit as st


def get_css_style():
    st.markdown(
        """
        <style>
        .stChatMessage-avatar {   
            border-radius: 50%; 
        }
        .stChatMessage-content {   
            word-break: break-word;
        }
        .st-emotion-cache-janbn0 {
            flex-direction: row-reverse;
            max-width: 50%;
            text-align: right;
            margin-left: auto; 
            margin-right: 0; 
            background-color: #424949;
        }
        .st-emotion-cache-4oy321 { 
            border-style: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_intro_html():
    st.markdown(
        """
        # 📝 Welcome to **EssayInsightsAI**

        #### Your AI powered Virtual Writing Tutor for writing smarter essays.

        ### 📥 Try It Now

    """
    )


def get_details_html():
    st.markdown("\n\n")
    st.markdown(
        """
            ### 🎯 What is EssayInsightsAI?

            EssayInsightAI is an intelligent web app that helps students and writers improve their essays by:
            - **Identifying key components** like Claims, Evidence, Leads, and Positions in the essay
            - **Providing personalized feedback** through an integrated AI chatbot
            - **Offering real-time insights** to enhance writing clarity, structure, and coherence

            Whether you're preparing for an exam or polishing an academic submission, EssayInsightAI guides you through each step of the writing process.

            ---

            ### 💡 What Can You Do Here?

            🔍 **Analyze Essays**  
            Break down your essay into its fundamental parts using our advanced NLP model trained specifically for student writing.

            🧠 **Get AI-Powered Feedback**  
            Ask our built-in chatbot to review your work, suggest improvements, and explain how to strengthen your arguments.

            ✏️ **Improve Your Writing**  
            Get actionable advice on grammar, clarity, structure, and more—instantly.

            ---

            ### 🚀 Why Use EssayInsightAI?

            ✅ Powered by state-of-the-art NLP & LLMs  
            ✅ Designed for student writing improvement  
            ✅ Easy-to-use interface with real-time results  
            ✅ Free and accessible—just paste your essay and go!

            ---
        """
    )
