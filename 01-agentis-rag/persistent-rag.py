from sqlitesearch import TextSearchIndex

sqlite_index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="faq.db",
)

import os

os.getcwd()

print(sqlite_index.count())

results = sqlite_index.search("Can I still join the course after it started?", num_results=99)

[doc["question"] for doc in results]

from rag_helper import RAGBase
from openai import OpenAI
from dotenv import load_dotenv


openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

load_dotenv()

assistant = RAGBase(
    index=sqlite_index,
    llm_client=openai_client,
)

answer = assistant.rag("Can i connect olama?")
print(answer)

sqlite_index.close()
