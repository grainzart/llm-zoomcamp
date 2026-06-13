import requests


docs_url = "https://datatalks.club/faq/json/courses.json"
response = requests.get(docs_url)
courses_raw = response.json()

documents = []
url_prefix = "https://datatalks.club/faq"

for course in courses_raw:
    course_url = f"{url_prefix}{course['path']}"

    course_response = requests.get(course_url)
    course_response.raise_for_status()
    course_data = course_response.json()

    documents.extend(course_data)

from pprint import pprint

pprint(documents[1111])

from minsearch import Index

index = Index(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
)

index.fit(documents)

# question = "capstone"


def search(question, course="llm-zoomcamp"):
    boost_dict = {"question": 2.0, "section": 0.5, "answer": 1.0}
    filter_dict = {"course": course}

    return index.search(
        question,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
        num_results=5,
    )


question = "I just discovered the course. Can I join now?"
search_results = search(question)

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""


def build_context(search_results):
    lines = []

    for doc in search_results:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt(question, search_results):
    context = build_context(search_results)
    prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )
    return prompt.strip()


prompt = build_prompt(question, search_results)

pprint(prompt)


# def rag(question):
#     search_results = search(question)
#     user_prompt = build_prompt(question, search_results)
#     return llm(user_prompt)
