import json

from llm import client, MODEL_NAME
from llm.lib.web import TOOLS, run_tool

MAX_TOOL_ROUNDS = 3

messages = [
    {
        "role": "system",
        "content": (
            "You are a casual, direct person. Keep replies short. "
            "No lists or headers unless the user explicitly asks for steps, a recipe, or a numbered list. "
            "Talk like a friend over text. "
            "Use search_web for recipes, current facts, or anything that should come from the internet. "
            "If they ask for recipes, search and give 3 options with titles and URLs. "
            "Use get_wikipedia when they want a Wikipedia article or a summary of one. "
            "Base those answers on the tool results, not memory."
        ),
    }
]


def assistant_to_message(assistant_message):
    # NIM rejects assistant messages with empty (or whitespace-only, once
    # trimmed) content -- a plain " " placeholder passes on the round it's
    # produced but gets rejected the next time it's replayed as history.
    payload = {
        "role": "assistant",
        "content": assistant_message.content or "...",
    }
    if assistant_message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in assistant_message.tool_calls
        ]
    return payload


print(f"Chat with {MODEL_NAME}. Type quit or exit to stop.\n")

while True:
    try:
        user_message = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not user_message or user_message.lower() in {"quit", "exit"}:
        break

    messages.append({"role": "user", "content": user_message})

    reply = ""
    for _ in range(MAX_TOOL_ROUNDS + 1):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_to_message(assistant_message))

        if not assistant_message.tool_calls:
            reply = (assistant_message.content or "").strip()
            break

        print("Looking that up...")
        for tool_call in assistant_message.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            result = run_tool(tool_call.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    print(f"Assistant: {reply}\n")
