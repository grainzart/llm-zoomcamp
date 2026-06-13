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
question = "I just discovered the course. Can I join now?"


def search(question, course="llm-zoomcamp"):
    boost_dict = {"question": 2.0, "section": 0.5, "answer": 1.0}
    filter_dict = {"course": course}

    return index.search(
        question,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
        num_results=5,
    )


search_results = search(question)


def rag(question):
    search_results = search(question)
    user_prompt = build_prompt(question, search_results)
    return llm(user_prompt)
