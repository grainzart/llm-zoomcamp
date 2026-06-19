from dotenv import load_dotenv
from ingest import load_faq_data, build_index
from rag_helper import RAGBase
from openai import OpenAI

load_dotenv()


documents = load_faq_data()
index = build_index(documents)

# openai_client = OpenAI()
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

custom_instructions = """
You're a course teaching assistant.
Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.
""".strip()

assistant = RAGBase(
    index=index,
    llm_client=openai_client,
    instructions=custom_instructions,
)

answer = assistant.rag("do you have certificates?")
assistant.rag("How do I get a certificate?")
assistant.rag("Can I still join the course after it started?")
