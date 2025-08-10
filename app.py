# app.py

import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- Page Configuration ---
st.set_page_config(page_title="HR Query Chatbot 🤖", page_icon="🤖", layout="wide")


# --- 1. Embedded Data & Pydantic Models ---
# The data is hardcoded here to keep the app self-contained.
EMPLOYEE_DATA = {
    "employees": [
        {"id": 1, "name": "Alice Johnson", "skills": ["Python", "React", "AWS", "Node.js"], "experience_years": 5, "projects": ["E-commerce Platform Migration", "Healthcare Dashboard UI"], "availability": "available"},
        {"id": 2, "name": "Dr. Sarah Chen", "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "Medical Data Processing"], "experience_years": 6, "projects": ["Medical Diagnosis Platform (Computer Vision)", "Genomic Data Analysis Pipeline"], "availability": "available", "notes": "Published 3 papers on healthcare AI."},
        {"id": 3, "name": "Michael Rodriguez", "skills": ["Python", "Machine Learning", "scikit-learn", "pandas", "HIPAA Compliance"], "experience_years": 4, "projects": ["Patient Risk Prediction System", "EHR Anonymization Tool"], "availability": "on project until 2025-10-15"},
        {"id": 4, "name": "David Smith", "skills": ["Java", "Spring Boot", "Microservices", "Kafka", "PostgreSQL"], "experience_years": 8, "projects": ["Financial Trading System Backend", "Banking API Gateway"], "availability": "available"},
        {"id": 5, "name": "Emily White", "skills": ["React Native", "JavaScript", "TypeScript", "Firebase", "GraphQL"], "experience_years": 3, "projects": ["Mobile Banking App", "Social Media Content App"], "availability": "available"},
        {"id": 6, "name": "Chris Green", "skills": ["DevOps", "Kubernetes", "Docker", "AWS", "Terraform", "CI/CD"], "experience_years": 7, "projects": ["Cloud Infrastructure Automation", "CI/CD Pipeline Optimization"], "availability": "available"},
        {"id": 7, "name": "Priya Patel", "skills": ["Data Science", "R", "SQL", "Tableau", "PowerBI"], "experience_years": 4, "projects": ["Customer Churn Analysis Dashboard", "Marketing Campaign ROI Prediction"], "availability": "available"},
        {"id": 8, "name": "Tom Clark", "skills": ["Go", "gRPC", "Prometheus", "System Design"], "experience_years": 6, "projects": ["High-performance Logging Service", "Real-time Bidding System"], "availability": "on project until 2025-11-01"},
        {"id": 9, "name": "Laura Martinez", "skills": ["UX/UI Design", "Figma", "Sketch", "User Research"], "experience_years": 5, "projects": ["Redesign of an e-learning platform", "User journey mapping for a fintech app"], "availability": "available"},
        {"id": 10, "name": "James Wilson", "skills": ["Python", "Django", "PostgreSQL", "Celery", "Redis"], "experience_years": 9, "projects": ["Scalable Web App for Logistics", "Content Management System"], "availability": "available"},
        {"id": 11, "name": "Zoe Brown", "skills": ["Cybersecurity", "Penetration Testing", "Metasploit", "Wireshark"], "experience_years": 5, "projects": ["Security Audit for a financial institution", "Network Vulnerability Assessment"], "availability": "on project until 2025-09-30"},
        {"id": 12, "name": "Ethan Hunt", "skills": ["React", "Next.js", "Vercel", "TailwindCSS"], "experience_years": 3, "projects": ["Corporate Website Overhaul", "Server-side Rendered Marketing Site"], "availability": "available"},
        {"id": 13, "name": "Grace Lee", "skills": ["Project Management", "Agile", "Scrum", "Jira"], "experience_years": 10, "projects": ["Led development of 'E-commerce Platform Migration'", "Coordinated 'Mobile Banking App' launch"], "availability": "available"},
        {"id": 14, "name": "Ben Carter", "skills": ["AWS", "Docker", "Python", "Bash Scripting", "Ansible"], "experience_years": 4, "projects": ["Automated cloud deployment scripts", "Containerization of legacy Java application"], "availability": "available"},
        {"id": 15, "name": "Olivia Garcia", "skills": ["Java", "Android", "Kotlin", "Jetpack Compose"], "experience_years": 4, "projects": ["Android App for a restaurant chain", "Fitness Tracking Mobile App"], "availability": "available"}
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


# --- 2. RAG System Logic ---
# The entire RAG system is now part of the Streamlit app script.
class RAGSystem:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API key is missing. Please add it to your Streamlit secrets.")
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

    def search(self, query: str, top_k: int = 3) -> Tuple[List[Employee], np.ndarray]:
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
            return f"An error occurred while communicating with the AI model: {e}"

    def generate_hr_response(self, query: str, context_employees: List[Employee]) -> str:
        context_str = "\n---\n".join([json.dumps(emp.model_dump()) for emp in context_employees])
        user_prompt = f"Based on the user query '{query}' and these employee profiles:\n{context_str}\n...generate a helpful and well-formatted recommendation."
        system_prompt = (
            "You are an intelligent HR assistant. Format your responses using Markdown. Use lists, bolding, "
            "and other elements to make the information clear and easy to read. For example, recommend candidates using a bulleted list."
        )
        return self._call_llm(user_prompt, system_prompt)

    def generate_general_response(self, query: str) -> str:
        system_prompt = (
            "You are a friendly and helpful conversational AI assistant. Use Markdown for formatting."
        )
        return self._call_llm(query, system_prompt)


# --- 3. Streamlit Application UI & Logic ---

# Use st.cache_resource to initialize the RAG system only once.
@st.cache_resource
def load_rag_system():
    """Loads and caches the RAG system and its components."""
    try:
        # For Streamlit Community Cloud, secrets are stored in st.secrets
        api_key = st.secrets.get("GROQ_API_KEY")
        return RAGSystem(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize the application: {e}")
        return None

def stream_response(text):
    """Simulates a streaming response for the chatbot word by word."""
    for word in text.split():
        yield word + " "
        time.sleep(0.04)

def display_employee_card(card_data):
    """Draws a visually appealing card for an employee."""
    with st.container(border=True):
        st.subheader(f"👤 {card_data['name']}")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**📅 Experience:** {card_data['experience_years']} years")
        with col2:
            st.success("**Status:** Available") if card_data['availability'] == 'available' else st.warning(f"**Status:** {card_data['availability']}")
        with st.expander("🛠️ View Skills & Projects"):
            st.write(f"**Skills:** `{', '.join(card_data['skills'])}`")
            st.write("**Past Projects:**")
            for project in card_data['projects']:
                st.markdown(f"- *{project}*")

# --- Main App ---
st.title("HR Resource Query Chatbot 🤖")
st.info("Find the right talent instantly by asking a question below.")

# Load the RAG system
rag_system = load_rag_system()

if rag_system:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "cards" in message and message["cards"]:
                for card in message["cards"]:
                    display_employee_card(card)

    # Process user input
    if prompt := st.chat_input("e.g., 'Find Python developers with AWS skills'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                # --- THIS IS THE NEW, DIRECT LOGIC ---
                # No more API calls. We call the RAG system methods directly.
                RELEVANCE_THRESHOLD = 1.0
                retrieved_employees, scores = rag_system.search(prompt)
                
                if retrieved_employees and scores[0][0] < RELEVANCE_THRESHOLD:
                    answer = rag_system.generate_hr_response(prompt, retrieved_employees)
                    cards_to_show = [emp.model_dump() for emp in retrieved_employees]
                else:
                    answer = rag_system.generate_general_response(prompt)
                    cards_to_show = []

                st.write_stream(stream_response(answer))
                if cards_to_show:
                    for card in cards_to_show:
                        display_employee_card(card)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer, "cards": cards_to_show})
else:
    st.warning("Application is not configured correctly. Please check secrets or logs.")
