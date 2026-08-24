import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# calendar.events covers both creating events and listing the ones already there.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CRED_FILE = os.path.join(_ROOT, "google-cred.json")
TOKEN_FILE = os.path.join(_ROOT, "token.json")


class CalendarNotAuthorized(RuntimeError):
    """Raised when the mirror tries to read the calendar before OAuth has run."""


def _get_calendar_service(*, interactive: bool = True):
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif interactive:
            flow = InstalledAppFlow.from_client_secrets_file(CRED_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        else:
            raise CalendarNotAuthorized(
                "Google Calendar is not authorized. Place google-cred.json in "
                "the repo root and sign in once so token.json can be created."
            )
        with open(TOKEN_FILE, "w") as token_file:
            token_file.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def _parse_google_time(value: dict, time_zone: str) -> tuple[datetime, bool]:
    """Return (aware datetime, is_all_day) from a Google start/end payload."""
    tz = ZoneInfo(time_zone)
    if "dateTime" in value:
        stamp = value["dateTime"].replace("Z", "+00:00")
        return datetime.fromisoformat(stamp).astimezone(tz), False
    return datetime.fromisoformat(value["date"]).replace(tzinfo=tz), True


def _week_end(now: datetime) -> datetime:
    """Midnight after Saturday of the current US week (Sunday–Saturday)."""
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    days_since_sunday = (now.weekday() + 1) % 7
    return start_of_today + timedelta(days=7 - days_since_sunday)


def list_upcoming_events(
    time_zone: str = "America/Los_Angeles",
    max_results: int = 20,
    calendar_id: str = "primary",
) -> list[dict]:
    """Events still happening today, plus anything else left this week."""
    service = _get_calendar_service(interactive=False)
    tz = ZoneInfo(time_zone)
    now = datetime.now(tz)
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    raw = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=start_of_today.isoformat(),
            timeMax=_week_end(now).isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=max_results,
        )
        .execute()
    )

    events = []
    for item in raw.get("items", []):
        if item.get("status") == "cancelled":
            continue
        start, all_day = _parse_google_time(item["start"], time_zone)
        end, _ = _parse_google_time(item["end"], time_zone)
        if end <= now:
            continue
        events.append(
            {
                "id": item.get("id"),
                "title": item.get("summary") or "(No title)",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "allDay": all_day,
                "location": item.get("location") or "",
            }
        )
    return events


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
