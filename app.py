# app.py

import os
import json
import time
from typing import List, Dict, Optional

import numpy as np
import faiss
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- 1. Page Configuration & UI Styling ---
st.set_page_config(page_title="Talent Finder AI", page_icon="🤖", layout="wide")

# Custom CSS for a sleek, modern interface
st.markdown("""
<style>
    /* General Styles */
    .stApp {
        background-color: #0d1117;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem;
    }
    /* Title */
    .title-text {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        background: -webkit-linear-gradient(45deg, #00FFA3, #00B8FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Employee Card Styling */
    .employee-card {
        border: 1px solid #2d333b;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        background-color: #161b22;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .employee-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 184, 255, 0.2);
        border-color: #00B8FF;
    }
    .employee-card h3 {
        color: #00B8FF;
        margin-top: 0;
    }
    .employee-card p {
        color: #c9d1d9;
        font-size: 0.95rem;
    }
    .employee-card summary {
        color: #8b949e;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. Data and Models ---

# NOTE: Your employee dictionary should be placed here.
# To keep the code clean, it's collapsed.
EMPLOYEE_DATA = {
  "employees": [
    { "id": 1, "name": "Alice Johnson", "skills": ["Python", "React", "AWS", "Node.js"], "experience_years": 5, "projects": ["E-commerce Platform Migration", "Healthcare Dashboard UI"], "availability": "available" },
    { "id": 2, "name": "Dr. Sarah Chen", "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "Medical Data Processing"], "experience_years": 6, "projects": ["Medical Diagnosis Platform (Computer Vision)", "Genomic Data Analysis Pipeline"], "availability": "available", "notes": "Published 3 papers on healthcare AI." },
    { "id": 3, "name": "Michael Rodriguez", "skills": ["Python", "Machine Learning", "scikit-learn", "pandas", "HIPAA Compliance"], "experience_years": 4, "projects": ["Patient Risk Prediction System", "EHR Anonymization Tool"], "availability": "on project until 2025-10-15" },
    { "id": 4, "name": "David Smith", "skills": ["Java", "Spring Boot", "Microservices", "Kafka", "PostgreSQL"], "experience_years": 8, "projects": ["Financial Trading System Backend", "Banking API Gateway"], "availability": "available" },
    { "id": 5, "name": "Emily White", "skills": ["React Native", "JavaScript", "TypeScript", "Firebase", "GraphQL"], "experience_years": 3, "projects": ["Mobile Banking App", "Social Media Content App"], "availability": "available" },
    { "id": 6, "name": "Chris Green", "skills": ["DevOps", "Kubernetes", "Docker", "AWS", "Terraform", "CI/CD"], "experience_years": 7, "projects": ["Cloud Infrastructure Automation", "CI/CD Pipeline Optimization"], "availability": "available" },
    { "id": 7, "name": "Priya Patel", "skills": ["Data Science", "R", "SQL", "Tableau", "PowerBI"], "experience_years": 4, "projects": ["Customer Churn Analysis Dashboard", "Marketing Campaign ROI Prediction"], "availability": "available" },
  ]
}

class Employee(BaseModel):
    id: int
    name: str
    skills: List[str]
    experience_years: int
    projects: List[str]
    availability: str
    notes: Optional[str] = None


# --- 3. RAG System (Backend Logic) ---
class RAGSystem:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API key is missing. Please add it to Streamlit secrets.")
        self.employees = EMPLOYEE_DATA['employees']
        self.employee_map = {emp['id']: emp for emp in self.employees}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = self._create_documents()
        self.index = self._create_faiss_index()
        self.llm_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

    def _create_documents(self) -> List[str]:
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

    def search(self, query: str, top_k: int = 3) -> tuple[List[Employee], np.ndarray]:
        query_embedding = self.embedding_model.encode([query])
        distances, ids = self.index.search(query_embedding, top_k)
        retrieved_employees = [Employee(**self.employee_map[eid]) for eid in ids[0] if eid != -1]
        return retrieved_employees, distances

    def _call_llm(self, user_prompt: str, system_prompt: str) -> str:
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
            st.error(f"LLM API Error: {e}")
            return "I'm sorry, I encountered an error while generating a response."

    def generate_hr_response(self, query: str, context_employees: List[Employee]) -> str:
        context_str = "\n---\n".join([json.dumps(emp.model_dump()) for emp in context_employees])
        user_prompt = f"Based on the user query '{query}' and these employee profiles:\n{context_str}\n...generate a helpful summary recommendation. Do not repeat all the details from the profiles, just provide a concise summary of why they are a good fit."
        system_prompt = "You are an intelligent HR assistant. Format your responses using Markdown. Use lists, bolding, and italics to make the information clear and readable."
        return self._call_llm(user_prompt, system_prompt)

    def generate_general_response(self, query: str) -> str:
        system_prompt = "You are a friendly and helpful conversational AI assistant. Use Markdown for all formatting."
        return self._call_llm(query, system_prompt)

@st.cache_resource
def load_rag_system():
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        return RAGSystem(api_key=api_key)
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return None


# --- 4. UI Helper Functions ---
def stream_response(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.04)

def display_employee_card(card_data: dict, container):
    # This function now uses native Streamlit components inside a passed container
    with container:
        st.markdown(
            f"""
            <div class="employee-card">
                <h3>👤 {card_data['name']}</h3>
                <p><b>📅 Experience:</b> {card_data['experience_years']} years</p>
                <p><b>📌 Status:</b> {'✅ Available' if card_data['availability'].lower() == 'available' else '⏳ ' + card_data['availability']}</p>
                <details>
                    <summary><b>🛠 Skills & Projects</b></summary>
                    <p><b>Skills:</b> {', '.join(card_data['skills'])}</p>
                    <p><b>Projects:</b></p>
                    <ul>{''.join(f"<li>{proj}</li>" for proj in card_data['projects'])}</ul>
                </details>
                {f"<p><b>📝 Notes:</b> {card_data['notes']}</p>" if card_data.get('notes') else ""}
            </div>
            """,
            unsafe_allow_html=True
        )

# --- 5. Main Application ---

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.markdown("This AI-powered chatbot helps HR teams find the right talent by answering natural language queries.")
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main Page ---
st.markdown('<h1 class="title-text">Talent Finder AI 🤖</h1>', unsafe_allow_html=True)

# Initialize RAG system
rag_system = load_rag_system()

if rag_system:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # The key change: Display cards within an expander from history
            if message.get("cards"):
                with st.expander("👥 View Recommended Candidate Profiles", expanded=False):
                    num_cards = len(message["cards"])
                    cols = st.columns(num_cards)
                    for i, card in enumerate(message["cards"]):
                        display_employee_card(card, cols[i])

    # Handle new user input
    if prompt := st.chat_input("e.g., 'Find developers with machine learning skills'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Searching talent..."):
                retrieved_employees, scores = rag_system.search(prompt)
                RELEVANCE_THRESHOLD = 1.0 # Adjust as needed

                if retrieved_employees and scores[0][0] < RELEVANCE_THRESHOLD:
                    answer = rag_system.generate_hr_response(prompt, retrieved_employees)
                    cards_to_show = [emp.model_dump() for emp in retrieved_employees]
                else:
                    answer = rag_system.generate_general_response(prompt)
                    cards_to_show = []

                st.write_stream(stream_response(answer))

                # The key change: Place new cards inside a new expander
                if cards_to_show:
                    with st.expander("👥 View Recommended Candidate Profiles", expanded=True):
                        num_cards = len(cards_to_show)
                        cols = st.columns(num_cards if num_cards > 0 else 1)
                        for i, card in enumerate(cards_to_show):
                            display_employee_card(card, cols[i])

                st.session_state.messages.append({"role": "assistant", "content": answer, "cards": cards_to_show})
else:
    st.warning("Application could not start. Please verify your API key in Streamlit secrets.")
