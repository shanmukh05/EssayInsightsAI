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
            max-width: 80%;
            text-align: right;
            margin-left: auto; 
            margin-right: 0; 
            background-color: #424949;
            border-radius: 25px; 
        }
        .st-emotion-cache-4oy321 { 
            border-style: none;
        }
        .stSidebar {
            min-width: 320px;
            max-width: 320px;
        }
        [data-testid="stFileUploaderDropzone"]{
            height: 133px;
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
            ### 👤 About Me
            I am Shanmukha Sainath, working as AI Engineer at KLA Corporation. I have done my Bachelors from Department of Electronics and Electrical Communication Engineering department with Minor in Computer Science Engineering and Micro in Artificial Intelligence and Applications from IIT Kharagpur.

            - Check my [ML Roadmap](https://github.com/shanmukh05/Machine-Learning-Roadmap) if you're interested to get into the field of ML
            - Check my [ScratchNLP](https://github.com/shanmukh05/scratch_nlp) Python library for understanding NLP algorithms implementation from scratch
            - [Connect with me](https://linktr.ee/shanmukh05) if you have any feedback or questions.

            ---

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

            ### 🚀 Why Use EssayInsightsAI?

            ✅ Powered by state-of-the-art NLP & LLMs  
            ✅ Designed for student writing improvement  
            ✅ Easy-to-use interface with real-time results  
            ✅ Free and accessible—just paste your essay and go!

            ---
        """
    )


def style_seg_button():
    st.markdown(
        """
        <style>
        .st-key-seg_button .stButton button {
            background-color: #4285F4;
            color: white;
            border: none;
            border-radius: 20px;
            padding: 6px 10px;
            font-size: 10px;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.2s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            position: fixed;
            bottom: 115px;
            right: 80px;
        }

        .st-key-seg_button .stButton button:hover {
            background-color: #357ae8;
            transform: translateY(-2px);
        }

        .st-key-seg_button .stButton button:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .st-key-seg_spinner .stSpinner {
            position: fixed;
            bottom: 115px;
            left: 400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
