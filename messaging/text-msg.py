from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib

from dotenv import load_dotenv
load_dotenv(".env.local")

# ==================== CONFIGURATION ====================
# 1. Your email credentials (example using Gmail)
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")  # App password or OAuth

# 2. The recipient's Gmail address
RECIPIENT_GMAIL = os.environ.get("RECIPIENT_GMAIL")
# =======================================================

# Use the recipient's gmail address directly
email_receiver = RECIPIENT_GMAIL

# Set up the message payload
msg = MIMEMultipart()
msg['From'] = SENDER_EMAIL
msg['To'] = email_receiver
msg['Subject'] = "Automated Email from Python Script"

# The actual text body
body = "Hello! This is a completely free automated email from my Python script."
msg.attach(MIMEText(body, 'plain'))

try:
    print("Connecting to secure server...")
    # Using Gmail's SMTP server settings
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()  # Upgrade the connection to secure encryption
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        print(f"Sending email to {email_receiver}...")
        server.send_message(msg)
        
    print("Success! The email has been sent.")

except Exception as e:
    print(f"\nAn error occurred while trying to send: {e}")
