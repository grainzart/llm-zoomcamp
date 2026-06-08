from dotenv import load_dotenv
import os

load_dotenv()

from openai import OpenAI

# openai_client = OpenAI()
os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# response = openai_client.models.list()
# print(response.model_dump_json(indent=2))


def llm(prompt):
    response = openai_client.responses.create(model="gpt-5.4-mini", input=prompt)
    return response.output_text


question = "ты знаешь алексея григорьева? программист и автор курсов по ии"
answer = llm(question)
print(answer)
