from langchain_community.retrievers import WikipediaRetriever


retriever = WikipediaRetriever()


docs = retriever.invoke("Machine Learning", doc_content_chars_max=100)

print("Document-1 : Information \n")
# print(docs[1].page_content)

print("\n")
# print(docs[1].metadata)
print(docs)
