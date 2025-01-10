# Function to format the Q-A pairs for the final prompt
def form_final_prompt(query, combined_qa_pair):

    # Extracting Q-A pairs and formatting them
    qa_text = "\n".join([f"Q: {qa_data['question']}\nA: {qa_data['answer']}" for qa in combined_qa_pair for qa_data in qa.values()])
    
    # Final prompt generation
    final_prompt = f"""
    Based on the following Q-A pairs, generate a concise, informative answer to the original query.

    Original Query: {query}

    Q-A Pairs:
    {qa_text}

    Using the above Q-A pairs, answer the question as accurately as possible. Ensure to reference specific verses and teachings from the Bhagavad Gita. Give a detailed answer:
    """
    
    return final_prompt

# Function to generate the final answer based on the combined Q-A pairs
def generate_final_answer(query, combined_qa_pair,model):

    final_prompt = form_final_prompt(query, combined_qa_pair)
    
    # Generate the final response from the model
    final_response = model.generate_content(final_prompt)
    
    return final_response.text.strip()
