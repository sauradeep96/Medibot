from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = download_hugging_face_embeddings()

index_name = "medibot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)


# ✅ Single prompt using system_prompt + context + question
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt + "\n\nContext:\n{context}"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    docs = retriever.invoke(msg)
    context = "\n".join([doc.page_content for doc in docs])
    response = chain.invoke({
        "context": context,
        "question": msg
    })
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)