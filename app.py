import streamlit as st
from indexing_retriever import split_documents, create_sparse_retriever, create_hybrid_retriever, create_dense_vector_store
from dataset_preprocessing import updated_documents
from config import configure_api_and_model
from decomposition import generate_sub_queries
from retriever import load_dense_vector_store, get_main_query_context, get_sub_query_context
from generator import generate_complete_qa_pair
from final_generation import generate_final_answer
import json
import os
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(
    page_title="Bhagavad Gita Q&A System",
    page_icon="📖",
    layout="wide",
)

# Initialize the model and embeddings once
load_dotenv()
api_key = os.getenv("GENAI_API_KEY")
instructor_embeddings, model = configure_api_and_model(api_key)  # Load the model once

# Document and retrieval setup
optimized_splits = split_documents(updated_documents)

# Creating and saving the dense vector store (FAISS), after executing for 1st time vector db will be created so comment this line
dense_vectordb = create_dense_vector_store(optimized_splits, instructor_embeddings)

# Loading the vector db
dense_vectordb = load_dense_vector_store("vector_database", instructor_embeddings)

sparse_retriever = create_sparse_retriever(optimized_splits)
hybrid_retriever = create_hybrid_retriever(dense_vectordb, sparse_retriever, alpha=0.5)

def process_query(query):
    sub_questions = [q.strip() for q in generate_sub_queries(query, model).text.strip().split("\n") if q.strip()]
    query_filtered_docs, query_context = get_main_query_context(query, hybrid_retriever)
    sub_query_context, sub_query_filtered_docs = get_sub_query_context(sub_questions, hybrid_retriever, query)
    combined_qa_pair = generate_complete_qa_pair(query, query_context, sub_questions, sub_query_filtered_docs, model)
    final_answer = generate_final_answer(query, combined_qa_pair, model)

    combined_docs = list(query_filtered_docs.values()) + list(sub_query_filtered_docs.values())
    flattened_combined_docs = [doc for sublist in combined_docs for doc in sublist]
    unique_verses = set()
    result = {
        "query": query,
        "generated_answer": final_answer,
        "verses_relevant_to_the_query": []
    }

    for doc in flattened_combined_docs:
        document_info = {
            "chapter": doc.metadata.get("chapter", "N/A"),
            "verse": doc.metadata.get("verse", "N/A"),
            "sanskrit": doc.page_content.split("\n")[0] if len(doc.page_content.split("\n")) > 0 else "N/A",
            "translation": doc.page_content.split("\n")[1] if len(doc.page_content.split("\n")) > 1 else "N/A",
        }
        unique_key = (document_info["chapter"], document_info["verse"])
        if unique_key not in unique_verses:
            unique_verses.add(unique_key)
            result["verses_relevant_to_the_query"].append(document_info)

    return result

# Sidebar
st.sidebar.title("📖 Context-Aware Religious Chatbot")
st.sidebar.markdown(
    """
    Welcome to the Context-Aware Religious Chatbot .  
    Enter your query to retrieve answers and relevant verses from the sacred text.
    
    **Features:**
    - 🎯 Precise answers to your query.
    - 🔍 Relevant verses with translations.
    - 📥 Download the results as JSON.

    """
)

# Header
st.title("📖 Context-Aware Religious Chatbot")
st.markdown(
    """
    This application uses AI to provide answers from the Bhagavad Gita and Patanjali Yog Sutras to your queries.    
    ---
    """
)

# User Input
query_input = st.text_area("🔍 Enter your query below:", height=100, placeholder="e.g., What does the Gita say about duty?")

# Process the query
if st.button("✨ Generate Answer"):
    if query_input:
        with st.spinner("Processing your query... please wait."):

            result = process_query(query_input)

        # Display Generated Answer
        st.subheader("🎯 Generated Answer:")
        st.success(result["generated_answer"])

        # Display Relevant Verses
        st.subheader("📜 Verses Relevant to the Query:")
        with st.expander("View Relevant Verses"):
            for verse in result["verses_relevant_to_the_query"]:
                st.markdown(f"""
                - **Chapter**: {verse['chapter']}
                - **Verse**: {verse['verse']}
                - **Sanskrit**: {verse['sanskrit']}
                - **Translation**: {verse['translation']}
                ---
                """)

        # Download Option
        json_result = json.dumps(result, ensure_ascii=False, indent=4)
        st.download_button(
            label="📥 Download Results as JSON",
            data=json_result,
            file_name="results.json",
            mime="application/json"
        )
    else:
        st.warning("⚠️ Please enter a query to proceed.")

# Footer
st.markdown("---")
st.markdown(
    """
    Developed by **Rohan Chanana**  
    ✨ Explore the timeless wisdom of the Bhagavad Gita and Patanjali Yog Sutras.
    """
)
