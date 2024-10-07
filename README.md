Overview
This project implements a Retrieval-Augmented Generation (RAG) application that combines web search, document embedding, and language model generation. It leverages the Tavily API for web search, FAISS for efficient similarity search, and Groq for generating responses based on retrieved documents.
Features

    Web Search: Utilizes the Tavily API to fetch relevant documents based on user queries.
    Document Embedding: Computes embeddings of documents using the SentenceTransformer model.
    Similarity Search: Employs FAISS to quickly find similar documents based on embeddings.
    Language Model Integration: Uses Groq's LLM to generate responses based on retrieved context.

Requirements
To run this application, you will need:

    Python 3.7 or higher
    Required libraries:
        faiss
        numpy
        requests
        sentence-transformers
        groq

You can install the required libraries using pip:

bash
pip install faiss-cpu numpy requests sentence-transformers groq

Setup

    API Keys: Obtain your API keys for Tavily and Groq. Replace the placeholders in the code with your actual keys:

    GROQ_API_KEY = "your_groq_api_key"
    TAVILY_API_KEY = "your_tavily_api_key"

Run the Application: Execute the script from your terminal.

bash
python your_script_name.py

Interact with the Application: You will be prompted to enter a question. Type your query and press Enter.
