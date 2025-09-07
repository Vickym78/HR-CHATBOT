# app.py

import json
import re
from typing import List, Optional, Set

import numpy as np
import faiss
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- 1. Streamlit Page Config & Styling ---
st.set_page_config(page_title="Talent Finder AI", page_icon="✨", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #c9d1d9; }
.title-text { 
    font-size: 3rem; 
    font-weight: 700; 
    text-align: center; 
    margin-bottom: 2rem; 
    background: -webkit-linear-gradient(45deg, #00FFA3, #00B8FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.employee-card { 
    border: 1px solid #2d333b; 
    border-radius: 12px; 
    padding: 20px;
    background-color: #161b22; 
    box-shadow: 0 4px 8px rgba(0,0,0,0.2); 
    margin-bottom: 15px; 
}
.employee-card h3 { color: #00B8FF; margin-top: 0; }
.employee-card p { color: #c9d1d9; font-size: 0.95rem; margin-bottom: 5px; }
.stChatInput > div { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

# --- 2. Static Data (Could be loaded from a DB or file) ---
EMPLOYEE_DATA = {
    "employees": [
        {"id": 1, "name": "Alice Johnson", "skills": ["Python", "React", "AWS", "Node.js", "SQL"], "experience_years": 5, "projects": ["E-commerce Platform Migration", "Healthcare Dashboard UI"], "availability": "available"},
        {"id": 2, "name": "Dr. Sarah Chen", "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "Keras"], "experience_years": 6, "projects": ["Medical Diagnosis Platform", "Genomic Data Analysis Pipeline"], "availability": "available", "notes": "Published 3 papers on healthcare AI."},
        {"id": 3, "name": "Michael Rodriguez", "skills": ["Python", "Machine Learning", "scikit-learn", "Pandas"], "experience_years": 4, "projects": ["Patient Risk Prediction System", "EHR Anonymization Tool"], "availability": "on project until 2025-10-15"},
        {"id": 4, "name": "David Smith", "skills": ["Java", "Spring Boot", "Kubernetes", "Docker", "GCP"], "experience_years": 8, "projects": ["Microservices Architecture for Fintech", "Cloud Deployment Automation"], "availability": "available"},
        {"id": 5, "name": "Emily White", "skills": ["JavaScript", "Vue.js", "Firebase", "UX/UI Design"], "experience_years": 3, "projects": ["Real-time Collaborative Whiteboard", "Social Media Analytics App"], "availability": "available"},
        {"id": 6, "name": "Chris Green", "skills": ["Go", "Python", "Terraform", "Ansible", "CI/CD"], "experience_years": 7, "projects": ["Infrastructure as Code for Banking", "Automated Security Auditing"], "availability": "on project until 2026-01-20"},
        {"id": 7, "name": "Priya Patel", "skills": ["Python", "Django", "PostgreSQL", "React"], "experience_years": 5, "projects": ["Supply Chain Management Portal", "Customer Relationship Manager (CRM)"], "availability": "available"}
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

# --- 3. Retrieval-Augmented Generation (RAG) System ---
class RAGSystem:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API key is missing. Please add it to your Streamlit secrets.")
        self.employees = [Employee(**e) for e in EMPLOYEE_DATA['employees']]
        self.employee_map = {e.id: e for e in self.employees}
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = self._create_documents()
        self.index = self._build_faiss_index()
        self.llm_client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.all_skills = {skill.lower() for emp in self.employees for skill in emp.skills}

    def _create_documents(self) -> List[str]:
        return [f"Name: {e.name}. Skills: {', '.join(e.skills)}. Experience: {e.experience_years} years. Key Projects: {', '.join(e.projects)}. Notes: {e.notes or 'N/A'}" for e in self.employees]

    def _build_faiss_index(self):
        embeddings = np.array(self.embedding_model.encode(self.documents, convert_to_tensor=False)).astype('float32')
        index = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings.shape[1]))
        index.add_with_ids(embeddings, np.array([e.id for e in self.employees]))
        return index

    def is_search_query(self, query: str) -> bool:
        """Uses an LLM to quickly classify if the user wants to find a candidate."""
        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Your task is to classify user intent. Does the following query ask to find a person, candidate, employee, or talent with specific skills or experience? Respond with only the word 'yes' or 'no'."},
                    {"role": "user", "content": query}
                ],
                max_tokens=3,
                temperature=0.0
            )
            decision = response.choices[0].message.content.strip().lower()
            return 'yes' in decision
        except Exception:
            # Fallback to simple keyword check if API fails
            keywords = ['find', 'developer', 'engineer', 'who has', 'experience', 'skills', 'years', 'candidate']
            return any(keyword in query.lower() for keyword in keywords)

    def general_chat(self, query: str) -> str:
        """Handles general conversation when not searching for candidates."""
        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a friendly and helpful assistant for the Talent Finder AI application. Your purpose is to help users find internal talent. Keep your responses concise and guide the user on how to ask for candidates (e.g., 'You can ask me to find developers with specific skills and experience.')"},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Could not connect to the chat model: {e}")
            return "Hello! I'm here to help you find talent. How can I assist you today?"

    def _filter_candidates(self, query: str) -> Set[int]:
        query_lower = query.lower()
        candidate_ids = set(self.employee_map.keys())
        match = re.search(r'(?:\d+\+?|\bat least \d+|\bmore than \d+)\s*years?', query_lower)
        if match:
            min_exp_match = re.search(r'\d+', match.group(0))
            if min_exp_match:
                min_exp = int(min_exp_match.group(0))
                candidate_ids &= {e.id for e in self.employees if e.experience_years >= min_exp}
        required_skills = {skill for skill in self.all_skills if skill in query_lower}
        if required_skills:
            candidate_ids &= {e.id for e in self.employees if all(req_skill in [s.lower() for s in e.skills] for req_skill in required_skills)}
        return candidate_ids

    def search(self, query: str, top_k: int = 5) -> List[Employee]:
        filtered_ids = self._filter_candidates(query)
        if not filtered_ids:
            return []
        query_vector = self.embedding_model.encode([query], convert_to_tensor=False).astype('float32')
        _, ids = self.index.search(query_vector, k=len(self.employees))
        ranked_results = []
        for emp_id in ids[0]:
            if emp_id in filtered_ids:
                ranked_results.append(self.employee_map[emp_id])
            if len(ranked_results) >= top_k:
                break
        return ranked_results

    def ask_llm(self, query: str, context: List[Employee]) -> str:
        context_str = "\n".join([json.dumps(e.model_dump()) for e in context])
        system_prompt = "You are an expert HR assistant AI. Your task is to analyze a user's query and a list of candidate profiles to provide a professional, concise, and helpful hiring recommendation. Start with a direct 1-2 sentence summary. Identify the **Top Recommendation** and explain why they are the best fit in bullet points. List **Other Suitable Candidates** with a brief one-sentence justification. Maintain a professional tone and use Markdown for formatting."
        user_prompt = f"<query>{query}</query>\n<candidate_profiles>{context_str}</candidate_profiles>\nPlease provide your hiring recommendation."
        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.5, max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception:
            return "Sorry, I encountered an error while generating the recommendation. Please try again."

@st.cache_resource
def load_rag_system():
    try:
        return RAGSystem(api_key=st.secrets["GROQ_API_KEY"])
    except (ValueError, KeyError) as e:
        st.error(f"Initialization failed: {e}. Please ensure GROQ_API_KEY is set in your Streamlit secrets.")
        return None

def display_employee(emp: Employee):
    availability_icon = "✅" if emp.availability == "available" else "⏳"
    availability_text = "Available" if emp.availability == "available" else emp.availability.replace("_", " ").title()
    st.markdown(f"""
    <div class="employee-card">
        <h3>👤 {emp.name}</h3>
        <p><b>Experience:</b> {emp.experience_years} years</p><p><b>Status:</b> {availability_icon} {availability_text}</p>
        <p><b>Skills:</b> {", ".join(emp.skills)}</p><p><b>Projects:</b> {", ".join(emp.projects)}</p>
        {f"<p><b>Notes:</b> {emp.notes}</p>" if emp.notes else ""}
    </div>""", unsafe_allow_html=True)

# --- 5. Main Streamlit App Flow ---
def main():
    st.markdown('<h1 class="title-text">Talent Finder AI ✨</h1>', unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.subheader("Controls")
        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        st.markdown("---")
        st.info("Developed by Vicky")

    rag_system = load_rag_system()
    if not rag_system:
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "employees" in message and message["employees"]:
                with st.expander("👥 Recommended Candidates", expanded=False):
                    for emp_data in message["employees"]:
                        display_employee(Employee(**emp_data))

    if user_query := st.chat_input("Find talent (e.g., 'Python dev with 5+ years experience')"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            # --- Intent Detection Step ---
            if rag_system.is_search_query(user_query):
                with st.spinner("Searching for candidates..."):
                    employees = rag_system.search(user_query)
                if employees:
                    with st.spinner("Generating recommendation..."):
                        llm_answer = rag_system.ask_llm(user_query, employees)
                    st.markdown(llm_answer)
                    with st.expander("👥 Recommended Candidates", expanded=True):
                        for emp in employees:
                            display_employee(emp)
                    st.session_state.messages.append({"role": "assistant", "content": llm_answer, "employees": [e.model_dump() for e in employees]})
                else:
                    no_results_message = "I couldn't find any candidates matching those specific criteria. You could try broadening your search."
                    st.write(no_results_message)
                    st.session_state.messages.append({"role": "assistant", "content": no_results_message, "employees": []})
            else:
                # --- General Chat Step ---
                with st.spinner("Thinking..."):
                    general_response = rag_system.general_chat(user_query)
                st.write(general_response)
                st.session_state.messages.append({"role": "assistant", "content": general_response})

if __name__ == "__main__":
    main()
