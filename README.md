# 📖 Context Aware Religious Chatbot

A Streamlit-based application that answers queries using the sacred texts of the **Bhagavad Gita** and **Patanjali Yog Sutras**. This chatbot combines advanced AI with hybrid retrieval techniques to deliver precise answers and relevant verses.

## 🌟 Features

- **AI-Powered Answers**: Generates contextual responses to your queries.
- **Verse Retrieval**: Displays related verses with translations.
- **Hybrid Search**: Combines dense (vector) and sparse (keyword) retrieval for accuracy.
- **Sub-Query Processing**: Decomposes queries for improved understanding.
- **Download Results**: Export answers and verses as a JSON file.

## 📂 Project Structure

- **`app.py`**: Main application script.
- **`indexing_retriever.py`**: Handles document splitting and retrieval mechanisms.
- **`decomposition.py`**: Decomposes user queries into sub-queries for better comprehension.
- **`dataset_preprocessing.py`**: Prepares the dataset for processing and optimization.
- **`retriever.py`**: Fetches query-related context and sub-query results.
- **`generator.py`**: Generates answers based on the retrieved context.
- **`final_generation.py`**: Combines answers into a final, cohesive response.
- **`config.py`**: Configures the AI models and embeddings (API key setup instructions are in this file).

## 📥 Dataset

Download the datasets:

- Bhagavad Gita dataset: [Bhagwad Gita Verses](https://github.com/atmabodha/Vedanta_Datasets/blob/main/Bhagwad_Gita/Bhagwad_Gita_Verses_English_Questions.csv)
- Patanjali Yog Sutras dataset: [Patanjali Yoga Sutras Verses](https://github.com/atmabodha/Vedanta_Datasets/blob/main/Patanjali_Yoga_Sutras/Patanjali_Yoga_Sutras_Verses_English_Questions.csv)
