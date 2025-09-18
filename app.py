import streamlit as st
import ollama
import chromadb
from pathlib import Path
import os

def load_cybersecurity_rules():
    seed_file = Path("./seed/rules.txt")
    if not seed_file.exists():
        return []
    text = seed_file.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines

documents = load_cybersecurity_rules()

client = chromadb.PersistentClient(path="./mydb/")
collection = client.get_or_create_collection(name="docs")

if collection.count() == 0:
    for i, doc in enumerate(documents):
        response = ollama.embed(model="nomic-embed-text", input=doc)
        embeddings = response["embeddings"]
        collection.add(
            ids=[str(i)],
            embeddings=embeddings,
            documents=[doc]
        )

def get_relevant_context(prompt):
    prompt_response = ollama.embed(model="nomic-embed-text", input=prompt)
    prompt_embedding = prompt_response["embeddings"]
    results = collection.query(
        query_embeddings=prompt_embedding,
        n_results=1
    )
    relevant_document = results['documents'][0][0] if results and 'documents' in results else None
    return relevant_document

st.title("Cyber Protector")
user_prompt = st.text_area("Enter a prompt to retrieve context:", height=200)

if st.button("Analyze Content"):
    context = get_relevant_context(user_prompt)
    if context:
        st.subheader("Relevant Context:")
        st.write(context)
        st.subheader("Generated Response:")
        response = ollama.generate(
            model="llama3:instruct",
            prompt=f"Using this data: {context}. Respond to this prompt: {user_prompt}\n"
                  f"{user_prompt}")
        st.write(response.get('response', 'No response generated'))
    else:
        st.write("No relevant context found.")