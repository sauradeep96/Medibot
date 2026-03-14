from dotenv import load_dotenv
import os

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# -------------------------
# Load environment variables
# -------------------------
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError(" PINECONE_API_KEY not found. Check your .env file.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

# -------------------------
# Set environment variables correctly
# -------------------------
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# -------------------------
# Load and process PDF data
# -------------------------
extracted_data = load_pdf_file(data='data/')
print(f"Pages loaded: {len(extracted_data)}")

if len(extracted_data) == 0:
    raise ValueError("No PDF data found. Make sure PDFs are inside the 'data/' folder.")

text_chunks = text_split(extracted_data)
print(f"Text chunks created: {len(text_chunks)}")

embeddings = download_hugging_face_embeddings()
print("Embeddings loaded.")

# -------------------------
# Initialize Pinecone client
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# -------------------------
# Create index if it doesn't exist
# -------------------------
index_name = "medibot"

if not pc.has_index(index_name):
    print(f"Creating Pinecone index: {index_name} ...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

index = pc.Index(index_name)

# -------------------------
# Store documents in Pinecone
# -------------------------
print("Uploading documents to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print(f"Successfully stored {len(text_chunks)} chunks in Pinecone index: '{index_name}'")
print("You can now run app.py!")