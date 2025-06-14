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

        ##### Your AI powered Virtual Writing Tutor for writing smarter essays.

        ⬇️ Scroll down to learn more about how to use this app and it's functionalities.
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

            ### 📚 How to Use EssayInsightsAI?

            1. **Start a New Session**: Click the "New Chat" button present in the side panel to begin a new analysis session.
            2. **Select Your Model**: Choose the appropriate model in the side panel for your essay analysis. The default is set to `gpt-4o`, which is optimized for conversational tasks and essay analysis.
            3. **Enter Your OpenAI API Key**: Input your OpenAI API key in the side panel to enable the AI functionalities. If you don't have one, you can sign up at [OpenAI API](https://openai.com/index/openai-api/).
            4. **Give your Essay**: You can either paste your essay text directly or upload a file.
            5. **Segmentation**: Click the "Segmentation" button to break down your essay into its following key components:
                - `Claim`: The main argument or thesis of your essay.
                - `Evidence`: Supporting facts, data, or examples that back up your claim.
                - `Lead`: The introduction or opening statement that sets the context for your essay.
                - `Position`: The stance or viewpoint you are taking in your essay.
                - `Counterclaim`: An opposing argument or viewpoint that you address in your essay.
                - `Rebuttal`: Your response to the counterclaim, defending your original position.
                - `Concluding Statement`: The final summary or closing argument that wraps up your essay.
                - `None`: If the segment does not fit into any of the above categories, it will be labeled as "None".
            6. **Chat with AI**: Ask questions about your essay, request feedback, or seek clarification on specific points.
            7. **Review Feedback**: Read the AI's suggestions and apply them to improve your essay.

            ---

            ### 🚀 Why Use EssayInsightsAI?

            ✅ Powered by state-of-the-art NLP & LLMs  
            ✅ Designed for student writing improvement  
            ✅ Easy-to-use interface with real-time results  
            ✅ Free and accessible—just paste your essay and go!

            ---

            ### 🙋🏻‍♂️ About Me
            I am Shanmukha Sainath, working as AI Engineer at KLA Corporation. I have done my Bachelors from Department of Electronics and Electrical Communication Engineering department with Minor in Computer Science Engineering and Micro in Artificial Intelligence and Applications from IIT Kharagpur.

            - Check my [ML Roadmap](https://github.com/shanmukh05/Machine-Learning-Roadmap) if you're interested to get into the field of ML
            - Check my [ScratchNLP](https://github.com/shanmukh05/scratch_nlp) Python library for understanding NLP algorithms implementation from scratch
            - [Connect with me](https://linktr.ee/shanmukh05) if you have any feedback or questions.

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
