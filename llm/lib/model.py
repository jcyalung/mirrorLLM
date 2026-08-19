from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv('.env.local')
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY"),
    # NIM 403s POST /v1/chat/completions when the User-Agent is the
    # stock openai-python client; any other UA is accepted.
    default_headers={"User-Agent": "mirrorLLM/1.0"},
)
MODEL_NAME = "google/diffusiongemma-26b-a4b-it"