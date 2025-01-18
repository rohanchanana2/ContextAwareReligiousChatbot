from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# Add your API key in a '.env' file

# Configure API and model
def configure_api_and_model(api_key):
    genai.configure(api_key=api_key)
    instructor_embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
    model = genai.GenerativeModel("gemini-1.5-flash")
    return instructor_embeddings, model
