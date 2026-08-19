import json
import os
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Import client and MODEL_NAME from the shared model library
from llm import client, MODEL_NAME

# 2. Google Calendar API Helper
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def get_calendar_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "google-cred.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def create_calendar_event(
    summary: str,
    start_iso: str,
    end_iso: str,
    description: str = "",
    time_zone: str = "America/Los_Angeles",
):
    service = get_calendar_service()
    event_body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_iso, "timeZone": time_zone},
        "end": {"dateTime": end_iso, "timeZone": time_zone},
    }
    created_event = (
        service.events().insert(calendarId="primary", body=event_body).execute()
    )
    return created_event.get("htmlLink")


# 3. Define the Tool / Function Schema for NIM
calendar_tool = {
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "Schedule a new event in Google Calendar.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "The title or subject of the meeting/event.",
                },
                "start_iso": {
                    "type": "string",
                    "description": "Start datetime in ISO 8601 format (YYYY-MM-DDTHH:MM:SS).",
                },
                "end_iso": {
                    "type": "string",
                    "description": "End datetime in ISO 8601 format (YYYY-MM-DDTHH:MM:SS).",
                },
                "description": {
                    "type": "string",
                    "description": "Optional notes, context, or meeting agenda.",
                },
            },
            "required": ["summary", "start_iso", "end_iso"],
        },
    },
}

# 4. Process Natural Language Request
user_input = (
    "Schedule a calendar event for next Monday; I have a doctor's appointment at 9:30"
    "Include a note to bring my ID."
)

current_context = (
    f"Today's date is {datetime.now().strftime('%A')}, and the date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
    "Timezone: America/Los_Angeles."
)

messages = [
    {
        "role": "system",
        "content": f"You are a calendar assistant. Use the provided tools to schedule events. Context: {current_context}",
    },
    {"role": "user", "content": user_input},
]

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=[calendar_tool],
    tool_choice="auto",
)

response_message = response.choices[0].message

# 5. Handle Tool Call Execution
if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        if tool_call.function.name == "create_calendar_event":
            args = json.loads(tool_call.function.arguments)
            print(f"Parsed Event Arguments: {json.dumps(args, indent=2)}")

            # Execute event creation
            link = create_calendar_event(
                summary=args.get("summary"),
                start_iso=args.get("start_iso"),
                end_iso=args.get("end_iso"),
                description=args.get("description", ""),
            )
            print(f"Event created successfully: {link}")
else:
    print("Model response:", response_message.content)