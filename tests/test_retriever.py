import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from app.retriever import get_retriever
r = get_retriever(k=4)

docs = r.invoke("Difference between Unsupervised and Supervised Learning")
print("Retrieved docs:", len(docs))
for d in docs[:3]:
    print(d.metadata, d.page_content[:300])