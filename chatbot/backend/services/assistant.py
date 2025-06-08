import openai
from openai import OpenAI

from chatbot.backend.utils.prompts import select_prompt_template


def generate_chat_reply(
    user_query,
    essay_text,
    essay_analysis,
    chat_history,
    model="gpt-4o",
    openai_key=None,
):
    openai.api_key = openai_key

    openai_client = OpenAI(
        api_key=openai.api_key,
    )

    prompt_template, prompt_type = select_prompt_template(user_query)
    chat_history = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in chat_history
        if msg["role"] in ["user", "assistant"]
    )

    if prompt_type == "analysis":
        prompt = prompt_template.format(
            essay_text=essay_text,
            essay_analysis=essay_analysis,
            specific_request=user_query,
        )
    elif prompt_type == "grammar":
        prompt = prompt_template.format(
            essay_text=essay_text,
            feedback_type=user_query,
        )
    else:
        prompt = prompt_template.format(
            chat_history=chat_history,
            current_question=user_query,
        )

    # Generate a response using OpenAI's API
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query},
        ],
        max_tokens=1500,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()
