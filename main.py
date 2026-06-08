from dotenv import load_dotenv
import os

load_dotenv()

from openai import OpenAI

openai_client = OpenAI()

print(os.getenv("OPENAI_API_KEY"))
