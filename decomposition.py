def generate_sub_queries(query,model):
    response = model.generate_content(
        f"""You are a helpful assistant specializing in decompositional reasoning. Your task is to break down a complex question into multiple specific sub-questions or sub-problems. These sub-questions should be:
        1. Specific and focused on a **single aspect** of the main question.
        2. **Non-overlapping** to avoid redundancy.
        3. Designed to help retrieve **highly relevant information** based on the original query. 

        **Input question**: {query}
        
        **Output (at least 3 distinct sub-queries):**
        1. [Sub-question 1]
        2. [Sub-question 2]
        3. [Sub-question 3]

        Ensure that the sub-queries are tightly focused, clear, and aimed at retrieving relevant documents based on the provided query.
        """
    )

    return response

