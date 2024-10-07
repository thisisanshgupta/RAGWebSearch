
import os
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from groq import Groq

# API Keys
GROQ_API_KEY = ""
TAVILY_API_KEY = ""

# Initialize SentenceTransformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Dimension of the sentence embeddings
index = faiss.IndexFlatL2(dimension)

class Document:
    def __init__(self, content: str, metadata: Dict):
        self.content = content
        self.metadata = metadata
        self.embedding = None

    def compute_embedding(self):
        self.embedding = model.encode([self.content])[0]

class RAGSearchEngine:
    def __init__(self):
        self.documents = []
        self.index = index

    def add_document(self, document: Document):
        document.compute_embedding()
        self.documents.append(document)
        self.index.add(np.array([document.embedding], dtype=np.float32))  # Ensure correct dtype

    def search(self, query: str, k: int = 5) -> List[Document]:
        query_embedding = model.encode([query])[0]
        _, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)  # Ensure correct dtype
        return [self.documents[i] for i in indices[0]]

class TavilySearch:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    def search(self, query: str) -> List[Dict]:
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "include_images": False,
            "max_results": 10
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            results = response.json().get("results", [])
            # Print URL with results
            for result in results:
                print(f"Results: {results}, URL: {self.base_url}")  # Print each result with the URL
            return results
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            if response is not None:
                print(f"Response content: {response.content}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request error occurred: {e}")
            raise


class GroqLLM:
    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = Groq(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.6) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=self.model_name,
            temperature=temperature
        )

        return chat_completion.choices[0].message.content


class RAGApplication:
    def __init__(self):
        self.search_engine = RAGSearchEngine()
        self.tavily_search = TavilySearch(TAVILY_API_KEY)
        self.llm = GroqLLM(GROQ_API_KEY)

    def process_query(self, query: str) -> str:
        # Step 1: Perform web search using Tavily
        search_results = self.tavily_search.search(query)

        # Step 2: Add search results to the RAG search engine
        for result in search_results:
            doc = Document(content=result["content"], metadata={"url": result["url"], "title": result["title"]})
            self.search_engine.add_document(doc)

        # Step 3: Retrieve relevant documents using the RAG search engine
        relevant_docs = self.search_engine.search(query, k=3)

        # Step 4: Prepare context for the LLM
        context = "\n\n".join([f"Title: {doc.metadata['title']}\nContent: {doc.content}" for doc in relevant_docs])

        # Step 5: Generate response using Groq LLM
        prompt = f"""Given the following context and question, provide a comprehensive and accurate answer. If the answer is not contained within the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""

        response = self.llm.generate(prompt)

        return response

# Example usage
if __name__ == "__main__":
    rag_app = RAGApplication()

    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        answer = rag_app.process_query(query)
        print("\nAnswer:", answer)
        print("\n" + "="*50 + "\n")
