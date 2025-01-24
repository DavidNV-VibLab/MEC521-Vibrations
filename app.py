import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from sympy import symbols, sqrt, pi, pretty
import os

# Load the embeddings and text chunks
embeddings = np.load("embeddings.npy")
with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n\n")

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_relevant_chunks(query, top_k=5):
    """
    Retrieve the top-k most relevant chunks for a given query.
    """
    # Embed the query
    query_embedding = model.encode([query])

    # Compute cosine similarity between the query and chunks
    similarities = cosine_similarity(query_embedding, embeddings)

    # Get the indices of the top-k most similar chunks
    top_k_indices = similarities.argsort()[0][-top_k:][::-1]

    # Retrieve the top-k chunks
    relevant_chunks = [chunks[i] for i in top_k_indices]
    return relevant_chunks

def generate_answer(query, relevant_chunks):
    """
    Generate an answer using Hugging Face's Inference API.
    """
    # Combine the relevant chunks into a single context
    context = "\n\n".join(relevant_chunks)

    # Call the Hugging Face API
    api_url = "https://api-inference.huggingface.co/models/distilbert-base-cased-distilled-squad"  # Smaller model
    headers = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE_API_KEY')}"}  # Use environment variable
    payload = {
        "inputs": {
            "question": query,
            "context": context
        }
    }

    # Retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            break  # Exit the loop if the request is successful
        elif response.status_code == 503:  # Model is loading
            error_data = response.json()
            estimated_time = error_data.get("estimated_time", 10)  # Default to 10 seconds
            print(f"Model is loading. Retrying in {estimated_time} seconds...")
            time.sleep(estimated_time)
        else:
            return f"Error: {response.status_code} - {response.text}"

    # Extract the answer
    if response.status_code == 200:
        answer = response.json().get("answer", "Sorry, I couldn't generate an answer.")
        return answer
    else:
        return f"Error: {response.status_code} - {response.text}"

# Streamlit app
st.title("MEC521 Vibrations Lecture Notes Q&A System")
st.write("Ask questions about your course lecture notes!")

# Input box for the user's question
query = st.text_input("Enter your question:")

if query:
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query)

    # Generate the answer
    answer = generate_answer(query, relevant_chunks)

    # Display the answer
    st.subheader("Answer:")
    st.write(answer)

    # Display the formula for natural frequency (example)
    k, m = symbols('k m')
    natural_frequency = sqrt(k / m) / (2 * pi)
    st.subheader("Formula for Natural Frequency:")
    st.latex(pretty(natural_frequency))
