from sentence_transformers import SentenceTransformer
from ingest import load_faq_data

documents = load_faq_data()
documents[10]

texts = []

for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)


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

query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)

results = vindex.search(
    query_vector,
    # filter_dict={"course": "llm-zoomcamp"},
    num_results=3,
)

print(results)
