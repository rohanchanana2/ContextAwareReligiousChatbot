from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever

# Function to split documents using RecursiveCharacterTextSplitter
def split_documents(documents, chunk_size=600, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return text_splitter.split_documents(documents)

# Function to create the FAISS vector store
def create_dense_vector_store(optimized_splits, instructor_embeddings):
    dense_vectordb = FAISS.from_documents(optimized_splits, instructor_embeddings)
    dense_vectordb.save_local("vector_database")
    return dense_vectordb

# Function to create the BM25 retriever
def create_sparse_retriever(optimized_splits):
    return BM25Retriever.from_documents(optimized_splits)

# Class for Custom Hybrid Retriever combining Dense and Sparse retrievers
class CustomHybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, alpha=0.5):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha

    def get_relevant_documents(self, query):
        dense_results = self.dense_retriever.get_relevant_documents(query)
        sparse_results = self.sparse_retriever.get_relevant_documents(query)

        score_dict = {}
        doc_mapping = {}

        for doc in dense_results:
            key = doc.page_content
            score_dict[key] = score_dict.get(key, 0) + self.alpha
            doc_mapping[key] = doc

        for doc in sparse_results:
            key = doc.page_content
            score_dict[key] = score_dict.get(key, 0) + (1 - self.alpha)
            doc_mapping[key] = doc

        sorted_keys = sorted(score_dict.keys(), key=lambda k: score_dict[k], reverse=True)
        return [doc_mapping[key] for key in sorted_keys]

# Function to create the CustomHybridRetriever instance
def create_hybrid_retriever(dense_vectordb, sparse_retriever, alpha=0.5):
    return CustomHybridRetriever(
        dense_retriever=dense_vectordb.as_retriever(),
        sparse_retriever=sparse_retriever,
        alpha=alpha
    )
