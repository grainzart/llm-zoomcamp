from sentence_transformers import SentenceTransformer

from ingest import load_faq_data


documents = load_faq_data()
documents[10]

texts = []

for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)

texts[10]
len(texts)


model = SentenceTransformer("all-MiniLM-L6-v2")

d = "You don't need to register. You're accepted.\
      You can also just start learning and submitting homework without registering."
dv = model.encode(d)


q1 = "Can I still join the course after the start date?"
v1 = model.encode(q1)

v1.dot(dv)

q2 = "How to install Docker on Windows?"
v2 = model.encode(q2)

v2.dot(dv)

from tqdm.auto import tqdm

batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i : i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)

# scores = []
# for i in range(len(vectors)):
#     score = v1.dot(vectors[i])
#     scores.append(score)

import numpy as np

x = np.array(vectors)
# x.shape
# print(x[:5, :3])

scores = x.dot(v1)

# arg max
idx = np.argmax(scores)
(idx, scores[idx])
documents[idx]

# top 5
top5_idx = np.argsort(scores)[-5:]
top5_idx = top5_idx[::-1]
scores[top5_idx]

# same with negative
top5_idx = np.argsort(-scores)[:5]

for i in top5_idx:
    print(scores[i])
    print(documents[i])
    print()
