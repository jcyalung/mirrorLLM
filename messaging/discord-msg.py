import json
import os
import urllib.request
from dotenv import load_dotenv
load_dotenv('.env.local')

def send_discord_notification(message_text):
    # Paste the exact Webhook URL you copied from Discord here
    WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
    
    # Format the data payload exactly how Discord expects it
    payload = {
        "content": message_text,
        "username": "Mirror Reminder" # You can override the bot name here
    }
    
    # Convert the payload dictionary into JSON bytes
    data = json.dumps(payload).encode('utf-8')
    
    # Configure the secure network request
    req = urllib.request.Request(
        WEBHOOK_URL, 
        data=data, 
        headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    )
    
    try:
        # Execute the post request
        with urllib.request.urlopen(req) as response:
            if response.status == 204: # Discord returns 204 No Content on success
                print("Notification sent to Discord successfully!")
            else:
                print(f"Unexpected response status: {response.status}")
                
    except Exception as e:
        print(f"Failed to send message: {e}")

# Test the function
send_discord_notification("Hello! This is a completely free alert from your Python script. 🚀")
