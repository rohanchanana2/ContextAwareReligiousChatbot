from langchain.document_loaders import CSVLoader
from langchain.schema import Document
import pandas as pd

bhagwad_Gita_dataset_path = 'Bhagwad_Gita_Verses_English_Questions.csv'
Patanjali_Yoga_Sutras_dataset_path = 'Patanjali_Yoga_Sutras_Verses_English_Questions.csv'

gita_df = pd.read_csv(bhagwad_Gita_dataset_path)
pys_df = pd.read_csv(Patanjali_Yoga_Sutras_dataset_path)

# Columns to drop
columns_to_drop = ["speaker"] 

# Dropping the specified columns
gita_df = gita_df.drop(columns=columns_to_drop)

gita_df["source"] = "Bhagwad Gita"
pys_df["source"] = "Patanjali Yoga Sutras"

# Cleaned dataset
cleaned_file_path = "./gita.csv"
gita_df.to_csv(cleaned_file_path, index=False)
cleaned_file_path = "./pys.csv"
pys_df.to_csv(cleaned_file_path, index=False)

# Loading the CSV files
gita_loader = CSVLoader(file_path="gita.csv", encoding="utf-8")
pys_loader = CSVLoader(file_path="pys.csv", encoding="utf-8")

# Loading the documents
gita_documents = gita_loader.load()
pys_documents = pys_loader.load()

# Combining documents from both sources
documents = gita_documents + pys_documents

updated_documents = []

# Updating metadata and simplifying page content for each document
for doc in documents:
    content_lines = doc.page_content.split('\n')
    chapter = content_lines[0].split(': ')[1]
    verse = content_lines[1].split(': ')[1]
    sanskrit = content_lines[2].split(': ')[1]
    translation = content_lines[3].split(': ')[1]
    source = content_lines[5].split(': ')[1]
    
    updated_metadata = {
        'source': source,
        'chapter': chapter,
        'verse': verse,
    }
    
    updated_page_content = f"Sanskrit: {sanskrit}\nTranslation: {translation}"
    
    updated_doc = Document(
        metadata=updated_metadata,
        page_content=updated_page_content
    )
    updated_documents.append(updated_doc)
