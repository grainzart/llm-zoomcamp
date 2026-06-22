import minsearch
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)


index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)
index.fit(documents)


INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:
    def __init__(self, index, llm_client, instructions=INSTRUCTIONS, prompt_template=PROMPT_TEMPLATE, course="llm-zoomcamp", model="gpt-5.4-mini"):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(query, num_results=num_results, boost_dict=boost_dict, filter_dict=filter_dict)

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        input_messages = [{"role": "developer", "content": self.instructions}, {"role": "user", "content": prompt}]

        response = self.llm_client.responses.create(model=self.model, input=input_messages)

        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer


class RAG:
    def __init__(self, index, llm_client, model="gpt-5.4-mini"):
        self.index = index
        self.llm_client = llm_client
        self.model = model

    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(f"File: {doc['filename']}")
            lines.append(doc["content"])
            lines.append("")
        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return PROMPT_TEMPLATE.format(question=query, context=context)

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.responses.create(model=self.model, input=input_messages)
        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response.output_text, response.usage.input_tokens


rag = RAG(index=index, llm_client=openai_client, model="gpt-5.4-mini")

query = "How does the agentic loop keep calling the model until it stops?"
answer, input_tokens = rag.rag(query)

print(f"Answer: {answer}")
print(f"---")
print(f"Input tokens: {input_tokens}")

from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
print(len(chunks))

import minsearch

# индекс из chunks
index_chunks = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)
index_chunks.fit(chunks)

rag_chunked = RAG(index=index_chunks, llm_client=openai_client, model="gpt-5.4-mini")

query = "How does the agentic loop keep calling the model until it stops?"
answer_chunked, input_tokens_chunked = rag_chunked.rag(query)

print(f"Input tokens (chunked): {input_tokens_chunked}")
print(f"Input tokens (before): 7126")
print(f"Ratio: {7126 / input_tokens_chunked:.1f}x")

from toyaikit.llm import OpenAIChatCompletionsClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIChatCompletionsRunner, DisplayingRunnerCallback


def search(query: str) -> dict[str, str]:
    """
    Search the course lessons for the given query.
    """
    return index_chunks.search(query, num_results=5)


agent_tools = Tools()
agent_tools.add_tool(search)

instructions = """
You're a course teaching assistant. Answer the student's question using the
search tool. Make multiple searches with different keywords before answering.
""".strip()

chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)

runner = OpenAIChatCompletionsRunner(
    tools=agent_tools,
    developer_prompt=instructions,
    chat_interface=chat_interface,
    llm_client=OpenAIChatCompletionsClient(
        model="gpt-5.4-mini",
        client=openai_client,
    ),
)

runner.run()
