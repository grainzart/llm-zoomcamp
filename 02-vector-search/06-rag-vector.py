from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

from ingest import load_faq_data, build_index

documents = load_faq_data()
index = build_index(documents)

from rag_helper import RAGBase

assistant = RAGBase(index=index, llm_client=openai_client)
query = "I just discovered the course. Can I still enroll?"

assistant.rag(query)


class RAGVector(RAGBase):
    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict,
        )


texts = []
for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

from tqdm.auto import tqdm

batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i : i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)

import numpy as np

x = np.array(vectors)

from minsearch import VectorSearch

vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(x, documents)

vector_assistant = RAGVector(
    embedder=model,
    index=vindex,
    llm_client=openai_client,
)

answer = vector_assistant.rag("the program has already begun, can I still sign up?")
print(answer)
