from langchain.vectorstores import FAISS

# Function to load the FAISS vector store
def load_dense_vector_store(index_path, embeddings):
    return FAISS.load_local(
        index_path, 
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

# Function to determine the query's context
def get_query_context(query):
    query_lower = query.lower()
    if "gita" in query_lower or "bhagwad gita" in query_lower:
        return "Bhagwad Gita"
    elif "patanjali yoga sutras" in query_lower or "pys" in query_lower:
        return "Patanjali Yoga Sutras"
    else:
        return "Unknown"

# Function to filter documents based on the context
def filter_documents_by_context(query_retrieved_docs, query):
    context = get_query_context(query)
    filtered_docs = {}

    for sub_question, docs in query_retrieved_docs.items():
        if context == "Bhagwad Gita":
            docs = [doc for doc in docs if "Bhagwad Gita" in doc.metadata.get('source', '')]
        elif context == "Patanjali Yoga Sutras":
            docs = [doc for doc in docs if "Patanjali Yoga Sutras" in doc.metadata.get('source', '')]
        elif context == "Unknown":
            docs = [doc for doc in docs if "Bhagwad Gita" in doc.metadata.get('source', '') or "Patanjali Yoga Sutras" in doc.metadata.get('source', '')]
        
        filtered_docs[sub_question] = docs

    return filtered_docs

# Function to retrieve relevant documents for the main query
def get_main_query_context(query, hybrid_retriever):
    query_retrieved_docs = hybrid_retriever.get_relevant_documents(query)
    query_retrieved_docs = {query: query_retrieved_docs}
    query_filtered_docs = filter_documents_by_context(query_retrieved_docs, query)

    # Combine the page content for the main query context
    return query_filtered_docs, "\n".join([doc.page_content for doc in query_filtered_docs[query]])

# Function to retrieve relevant documents for sub-queries
def get_sub_query_context(sub_questions, hybrid_retriever, query):
    sub_query_retrieved_docs = {}
    for sub_question in sub_questions:
        sub_query_retrieved_docs[sub_question] = hybrid_retriever.get_relevant_documents(sub_question)

    # Filter sub-query documents by context
    sub_query_filtered_docs = filter_documents_by_context(sub_query_retrieved_docs, query)
    
    return "\n".join([doc.page_content for doc in sub_query_filtered_docs.get(sub_question, [])]), sub_query_filtered_docs

