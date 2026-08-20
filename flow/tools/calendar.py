import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
CRED_FILE = "google-cred.json"
TOKEN_FILE = "token.json"


def _get_calendar_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CRED_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token_file:
            token_file.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def create_calendar_event(
    summary: str,
    start_iso: str,
    end_iso: str,
    description: str = "",
    time_zone: str = "America/Los_Angeles",
) -> str:
    """Create an event on the user's primary Google Calendar. Returns the event link."""
    service = _get_calendar_service()
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


CALENDAR_TOOL = {
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "Schedule a new event on the user's Google Calendar.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Title or subject of the event.",
                },
                "start_iso": {
                    "type": "string",
                    "description": "Start datetime, ISO 8601 (YYYY-MM-DDTHH:MM:SS).",
                },
                "end_iso": {
                    "type": "string",
                    "description": "End datetime, ISO 8601 (YYYY-MM-DDTHH:MM:SS).",
                },
                "description": {
                    "type": "string",
                    "description": "Optional notes, context, or agenda.",
                },
            },
            "required": ["summary", "start_iso", "end_iso"],
        },
    },
}
