from llm import client, MODEL_NAME

def get_llm_response(user_message: str, system_prompt: str = None):
    """
    Generate a response from the LLM using the NVIDIA NIM client.

    :param user_message: The user's message or prompt.
    :param system_prompt: (Optional) The system prompt or context for the conversation.
    :return: The assistant's reply as a string.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    return response.choices[0].message.content.strip()
  

user_message = """How do cook hard boiled eggs?"""
message = get_llm_response(user_message)
print(message)