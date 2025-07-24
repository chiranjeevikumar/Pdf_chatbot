from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
def create_faiss_index(texts: List[str]):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts,embeddings)
def retrieve_relevant_docs(vectorstores,query: str  , k: int = 5):
    docs = vectorstores.similarity_search(query,k=k)
    return docs