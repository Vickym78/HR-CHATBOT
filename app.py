import os
import json
import time
from typing import List, Optional

import numpy as np
import faiss
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- Streamlit Page Config ---
st.set_page_config(page_title="HR Query Chatbot 🤖", page_icon="🤖", layout="wide")

# --- Custom CSS for Cool UI ---
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
header, .stMarkdown {
    color: white !important;
}
.nameplate {
    background: linear-gradient(90deg, #ff4d4d, #ff884d);
    color: white;
    padding: 12px 25px;
    font-size: 28px;
    font-weight: bold;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(255, 77, 77, 0.5);
    margin-bottom: 25px;
}
.chat-bubble-user {
    background-color: #262730;
    padding: 10px 15px;
    border-radius: 12px;
    margin: 8px 0;
    color: #fff;
    max-width: 80%;
}
.chat-bubble-assistant {
    background-color: #1e1e2f;
    padding: 10px 15px;
    border-radius: 12px;
    margin: 8px 0;
    color: #d4d4d4;
    max-width: 80%;
}
.employee-card {
    border: 2px solid #4CAF50;
    border-radius: 12px;
    padding: 15px;
    margin: 12px 0;
    background-color: #1b1f24;
    box-shadow: 0 0 15px rgba(76, 175, 80, 0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.employee-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 25px rgba(76, 175, 80, 0.4);
}
</style>
""", unsafe_allow_html=True)

# --- Embedded Employee Data ---
EMPLOYEE_DATA = {...}  # same employee dictionary you already have

class Employee(BaseModel):
    id: int
    name: str
    skills: list
    experience_years: int
    projects: list
    availability: str
    notes: Optional[str] = None

# --- RAG System ---
class RAGSystem:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API key is missing.")
        self.employees = EMPLOYEE_DATA['employees']
        self.employee_map = {emp['id']: emp for emp in self.employees}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = self._create_documents()
        self.index = self._create_faiss_index()
        self.llm_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

    def _create_documents(self):
        return [
            f"Name: {emp['name']}. Experience: {emp['experience_years']} years. "
            f"Skills: {', '.join(emp['skills'])}. Past Projects: {', '.join(emp['projects'])}. "
            f"Notes: {emp.get('notes', 'N/A')}"
            for emp in self.employees
        ]

    def _create_faiss_index(self):
        embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=False)
        index = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings.shape[1]))
        ids = np.array([emp['id'] for emp in self.employees])
        index.add_with_ids(embeddings, ids)
        return index

    def search(self, query: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode([query])
        distances, ids = self.index.search(query_embedding, top_k)
        retrieved_employees = [Employee(**self.employee_map[eid]) for eid in ids[0] if eid != -1]
        return retrieved_employees, distances

    def _call_llm(self, user_prompt: str, system_prompt: str):
        try:
            response = self.llm_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {e}"

    def generate_hr_response(self, query: str, context_employees: List[Employee]):
        context_str = "\n---\n".join([json.dumps(emp.model_dump()) for emp in context_employees])
        user_prompt = f"Based on the query '{query}' and these profiles:\n{context_str}\nRecommend suitable candidates."
        system_prompt = "You are an intelligent HR assistant. Use Markdown formatting."
        return self._call_llm(user_prompt, system_prompt)

    def generate_general_response(self, query: str):
        system_prompt = "You are a friendly and helpful AI assistant. Use Markdown."
        return self._call_llm(query, system_prompt)

# --- Cache the RAG system ---
@st.cache_resource
def load_rag_system():
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        return RAGSystem(api_key=api_key)
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return None

# --- Helper functions ---
def stream_response(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.04)

def display_employee_card(card_data: dict):
    st.markdown(
        f"""
        <div class="employee-card">
            <h3>👤 {card_data['name']}</h3>
            <p><b>📅 Experience:</b> {card_data['experience_years']} years</p>
            <p><b>📌 Status:</b> {'✅ Available' if card_data['availability'].lower() == 'available' else '⏳ ' + card_data['availability']}</p>
            <details>
                <summary><b>🛠 Skills & Projects</b></summary>
                <p><b>Skills:</b> {', '.join(card_data['skills'])}</p>
                <ul>{''.join(f"<li>{proj}</li>" for proj in card_data['projects'])}</ul>
            </details>
            {f"<p><b>📝 Notes:</b> {card_data['notes']}</p>" if card_data.get('notes') else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Main App ---
st.markdown('<div class="nameplate">HR Resource Query Chatbot 🤖</div>', unsafe_allow_html=True)
st.info("Find the right talent instantly by asking a question below.")

rag_system = load_rag_system()

if rag_system:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-assistant'>{message['content']}</div>", unsafe_allow_html=True)
            if "cards" in message:
                for card in message["cards"]:
                    display_employee_card(card)

    if prompt := st.chat_input("e.g., 'Find Python developers with AWS skills'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='chat-bubble-user'>{prompt}</div>", unsafe_allow_html=True)

        retrieved_employees, scores = rag_system.search(prompt)
        RELEVANCE_THRESHOLD = 1.0

        if retrieved_employees and scores[0][0] < RELEVANCE_THRESHOLD:
            answer = rag_system.generate_hr_response(prompt, retrieved_employees)
            cards_to_show = [emp.model_dump() for emp in retrieved_employees]
        else:
            answer = rag_system.generate_general_response(prompt)
            cards_to_show = []

        st.write_stream(stream_response(answer))
        for card in cards_to_show:
            display_employee_card(card)

        st.session_state.messages.append({"role": "assistant", "content": answer, "cards": cards_to_show})
else:
    st.warning("App is not configured correctly. Please check your API key.")
