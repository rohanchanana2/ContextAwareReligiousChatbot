# Function to generate the answer for the main query
def generate_main_query_answer(query, query_context,model):
    prompt = f"""
    You are tasked with answering the following query based on the provided documents. 
    Your answer should reflect the teachings from the specific chapters and verses referenced 
    and stay true to the context provided in the documents. While the response should remain grounded 
    in the text, you can provide a generalized, thoughtful answer. Please answer the question using insights 
    drawn from the provided documents. Your answer should be coherent and focused on the teachings relevant to the question. 

    CONTEXT: {query_context} 

    QUESTION: {query}
    """
    
    response = model.generate_content(prompt)
    
    return response.text.strip()

# Function to create a prompt for each sub-question
def form_sub_question_prompt(sub_question, docs):
    document_text = "\n".join([f"Source: {doc.metadata['source']}, Chapter: {doc.metadata['chapter']}, Verse: {doc.metadata['verse']}\nContent: {doc.page_content[:500]}" for doc in docs])
    
    prompt = f"""
    You are tasked with answering the following query based on the provided documents. 
    Your answer should reflect the teachings from the specific chapters and verses referenced 
    and stay true to the context provided in the documents. While the response should remain grounded 
    in the text, you can provide a generalized, thoughtful answer. Please answer the question using insights 
    drawn from the provided documents. Your answer should be coherent and focused on the teachings relevant to the question.

    Sub-question: {sub_question}

    Relevant Documents:
    {document_text}
    """
    
    return prompt

# Function to generate answers for all sub-queries
def generate_sub_query_answers(sub_questions, sub_query_filtered_docs,model):
    sub_query_qa_pair = {}

    for sub_question, docs in sub_query_filtered_docs.items():
        prompt = form_sub_question_prompt(sub_question, docs)
        
        response = model.generate_content(prompt)
        
        sub_query_qa_pair[sub_question] = {
            "question": sub_question,
            "answer": response.text.strip()
        }
    
    return sub_query_qa_pair

# Function to generate the complete QA pair for both the main query and sub-queries
def generate_complete_qa_pair(query, query_context, sub_questions, sub_query_filtered_docs,model):
    # Generate answer for the main query
    query_qa_pair = {}
    query_qa_pair[query] = {
        "question": query,
        "answer": generate_main_query_answer(query, query_context,model)
    }

    # Generate answers for all sub-queries
    sub_query_qa_pair = generate_sub_query_answers(sub_questions, sub_query_filtered_docs,model)

    combined_qa_pair = [query_qa_pair, sub_query_qa_pair]

    return combined_qa_pair

