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

len(documents)

import minsearch

index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)

index.fit(documents)

query = "How does the agentic loop keep calling the model until it stops?"
results = index.search(query=query, num_results=5)

for n in results:
    print(n["filename"])
