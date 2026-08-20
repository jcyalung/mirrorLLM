import json
import os
import smtplib
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv(".env.local")


def notify_discord(message: str) -> dict:
    """Post a message to the configured Discord webhook."""
    webhook_url = os.environ.get("WEBHOOK_URL")
    if not webhook_url:
        return {"status": "error", "detail": "WEBHOOK_URL is not set."}

    payload = {"content": message, "username": "Mirror Assistant"}
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "mirrorLLM/1.0"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            if response.status == 204:
                return {"status": "sent", "preview": message[:60]}
            return {"status": "error", "detail": f"Unexpected status {response.status}"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def send_email(subject: str, body: str, recipient: str = None) -> dict:
    """Send a plain-text email via Gmail SMTP."""
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("SENDER_PASSWORD")
    to_addr = recipient or os.environ.get("RECIPIENT_GMAIL")

    if not sender or not password or not to_addr:
        return {
            "status": "error",
            "detail": "Email is not configured (SENDER_EMAIL/SENDER_PASSWORD/RECIPIENT_GMAIL).",
        }

    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = to_addr
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(message)
        return {"status": "sent", "to": to_addr, "preview": body[:60]}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


NOTIFY_DISCORD_TOOL = {
    "type": "function",
    "function": {
        "name": "notify_discord",
        "description": "Post a short notification message to the household Discord channel.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The text to post."},
            },
            "required": ["message"],
        },
    },
}

SEND_EMAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send a plain-text email, e.g. to forward a recipe, link, or note.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Email body text."},
                "recipient": {
                    "type": "string",
                    "description": "Recipient email address. Defaults to the configured recipient if omitted.",
                },
            },
            "required": ["subject", "body"],
        },
    },
}
