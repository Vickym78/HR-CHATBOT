# app.py

import json
import re
import time
from typing import List, Dict, Optional, Set

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
.stApp { background-color: #0d1117; }
.title-text { font-size: 3rem; font-weight: 700; text-align: center; 
  margin-bottom: 1rem; background: -webkit-linear-gradient(45deg, #00FFA3, #00B8FF);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.employee-card { border: 1px solid #2d333b; border-radius: 12px; padding: 20px;
  background-color: #161b22; box-shadow: 0 4px 8px rgba(0,0,0,0.2); margin-bottom: 10px; }
.employee-card h3 { color: #00B8FF; margin-top: 0; }
.employee-card p { color: #c9d1d9; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. Static Data (Employee Profiles) ---
EMPLOYEE_DATA = {
    "employees": [
        {"id": 1, "name": "Alice Johnson", "skills": ["Python", "React", "AWS", "Node.js"],
         "experience_years": 5, "projects": ["E-commerce Platform Migration", "Healthcare Dashboard UI"],
         "availability": "available"},
        {"id": 2, "name": "Dr. Sarah Chen", "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch"],
         "experience_years": 6, "projects": ["Medical Diagnosis Platform", "Genomic Data Analysis Pipeline"],
         "availability": "available", "notes": "Published 3 papers on healthcare AI."},
        {"id": 3, "name": "Michael Rodriguez", "skills": ["Python", "Machine Learning", "scikit-learn"],
         "experience_years": 4, "projects": ["Patient Risk Prediction System", "EHR Anonymization Tool"],
         "availability": "on project until 2025-10-15"},
        # ... add the rest of employees here
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

# --- 3. Retrieval-Augmented Generation System ---
class RAGSystem:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API key missing.")

        self.employees = EMPLOYEE_DATA['employees']
        self.employee_map = {e['id']: e for e in self.employees}
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Precompute embeddings once
        self.documents = [
            f"Name: {e['name']}. Skills: {', '.join(e['skills'])}. "
            f"Experience: {e['experience_years']} years. Projects: {', '.join(e['projects'])}. "
            f"Notes: {e.get('notes','N/A')}" for e in self.employees
        ]
        embeddings = np.array(self.embedding_model.encode(self.documents, convert_to_tensor=False))
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings.shape[1]))
        self.index.add_with_ids(embeddings, np.array([e['id'] for e in self.employees]))

        self.llm_client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.all_skills = {s.lower() for e in self.employees for s in e['skills']}

    def _filter_ids(self, query: str) -> Set[int]:
        """Extract skill + experience filters from query."""
        q = query.lower()
        ids = set(self.employee_map.keys())

        # Experience filter
        match = re.search(r'(\d+)\+?\s*years', q)
        if match:
            min_exp = int(match.group(1))
            ids &= {e['id'] for e in self.employees if e['experience_years'] >= min_exp}

        # Skill filter
        skills = {s for s in self.all_skills if s in q}
        if skills:
            ids &= {e['id'] for e in self.employees if all(s in [sk.lower() for sk in e['skills']] for s in skills)}

        return ids

    def search(self, query: str) -> List[Employee]:
        """Hybrid semantic + filter search."""
        filtered_ids = self._filter_ids(query)
        if not filtered_ids:
            return []

        query_vec = np.array(self.embedding_model.encode([query], convert_to_tensor=False))
        _, ids = self.index.search(query_vec, len(self.employees))

        seen, results = set(), []
        for eid in ids[0]:
            if eid in filtered_ids and eid not in seen:
                results.append(Employee(**self.employee_map[eid]))
                seen.add(eid)
        return results

    def ask_llm(self, query: str, context: List[Employee]) -> str:
        """Generate HR-friendly recommendation text."""
        ctx = "\n".join([json.dumps(e.model_dump()) for e in context])
        sys_prompt = "You are an expert HR assistant helping recruiters shortlist talent."
        user_prompt = f"Query: {query}\n\nCandidate Profiles:\n{ctx}\n\nGive a professional HR recommendation."

        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # ✅ updated model
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6
            )
            return response.choices[0].message["content"]
        except Exception as e:
            # Automatic fallback to 70B if instant fails
            try:
                response = self.llm_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.6
                )
                return response.choices[0].message["content"]
            except Exception as e2:
                return f"(LLM error: {e2})"

@st.cache_resource
def load_rag_system():
    return RAGSystem(api_key=st.secrets["GROQ_API_KEY"])

# --- 4. UI Helpers ---
def display_employee(emp: dict):
    st.markdown(f"""
    <div class="employee-card">
        <h3>👤 {emp['name']}</h3>
        <p><b>Experience:</b> {emp['experience_years']} years</p>
        <p><b>Status:</b> {"✅ Available" if emp['availability']=="available" else "⏳ "+emp['availability']}</p>
        <p><b>Skills:</b> {", ".join(emp['skills'])}</p>
        <p><b>Projects:</b> {", ".join(emp['projects'])}</p>
        {f"<p><b>Notes:</b> {emp['notes']}</p>" if emp.get("notes") else ""}
    </div>
    """, unsafe_allow_html=True)

# --- 5. Streamlit App Flow ---
with st.sidebar:
    st.subheader("🤖 Talent Finder AI")
    if st.button("Clear Chat"):
        st.session_state.clear()
        st.rerun()

st.markdown('<h1 class="title-text">Talent Finder AI ✨</h1>', unsafe_allow_html=True)

rag = load_rag_system()
if not rag:
    st.error("Could not initialize RAG System. Check API key.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# Chat Input
user_query = st.chat_input("Find talent (e.g. Python devs with 5+ years)...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        employees = rag.search(user_query)
        if employees:
            answer = rag.ask_llm(user_query, employees)
            st.write(answer)
            with st.expander("👥 Recommended Candidates", expanded=True):
                for emp in employees:
                    display_employee(emp.model_dump())
        else:
            st.write("I couldn't find any candidates. Try broadening your query.")

    st.session_state.history.append({"q": user_query, "res": employees})
