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
# For a real application, this data would come from a database, CSV, or an internal API.
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
        
        # Load a high-quality sentence transformer model for embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Create a textual representation of each employee for semantic search
        self.documents = self._create_documents()
        
        # Create a FAISS index for efficient similarity search
        self.index = self._build_faiss_index()

        # Initialize the Groq LLM client
        self.llm_client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        
        # Pre-calculate all unique skills for faster filtering
        self.all_skills = {skill.lower() for emp in self.employees for skill in emp.skills}

    def _create_documents(self) -> List[str]:
        """Creates a detailed string for each employee to be embedded."""
        return [
            f"Name: {e.name}. Skills: {', '.join(e.skills)}. "
            f"Experience: {e.experience_years} years. Key Projects: {', '.join(e.projects)}. "
            f"Notes: {e.notes or 'N/A'}"
            for e in self.employees
        ]

    def _build_faiss_index(self):
        """Encodes documents and builds the FAISS index."""
        embeddings = np.array(self.embedding_model.encode(self.documents, convert_to_tensor=False)).astype('float32')
        dimension = embeddings.shape[1]
        
        # IndexFlatL2 is a simple L2 distance search. IndexIDMap maps the internal FAISS index to our employee IDs.
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        index.add_with_ids(embeddings, np.array([e.id for e in self.employees]))
        return index

    def _filter_candidates(self, query: str) -> Set[int]:
        """Extracts hard filters (skills, experience) from the query to create a candidate pool."""
        query_lower = query.lower()
        candidate_ids = set(self.employee_map.keys())

        # Experience filter (e.g., "5+ years", "at least 3 years")
        match = re.search(r'(?:\d+\+?|\bat least \d+|\bmore than \d+)\s*years?', query_lower)
        if match:
            min_exp_match = re.search(r'\d+', match.group(0))
            if min_exp_match:
                min_exp = int(min_exp_match.group(0))
                candidate_ids &= {e.id for e in self.employees if e.experience_years >= min_exp}

        # Skill filter
        required_skills = {skill for skill in self.all_skills if skill in query_lower}
        if required_skills:
            candidate_ids &= {
                e.id for e in self.employees 
                if all(req_skill in [s.lower() for s in e.skills] for req_skill in required_skills)
            }
        
        return candidate_ids

    def search(self, query: str, top_k: int = 5) -> List[Employee]:
        """Performs a hybrid search: filter first, then rank by semantic similarity."""
        filtered_ids = self._filter_candidates(query)
        if not filtered_ids:
            return []

        # Encode the user's query
        query_vector = self.embedding_model.encode([query], convert_to_tensor=False).astype('float32')
        
        # Search the FAISS index to get semantically similar candidates
        _, ids = self.index.search(query_vector, k=len(self.employees))
        
        # Rank the filtered candidates by semantic similarity score
        ranked_results = []
        for emp_id in ids[0]:
            if emp_id in filtered_ids:
                ranked_results.append(self.employee_map[emp_id])
            if len(ranked_results) >= top_k:
                break
        
        return ranked_results

    def ask_llm(self, query: str, context: List[Employee]) -> str:
        """Generates a professional, HR-friendly recommendation using an optimized prompt."""
        context_str = "\n".join([json.dumps(e.model_dump()) for e in context])
        
        # This optimized prompt uses clear instructions and XML-like tags to guide the LLM
        # for a more structured and reliable output.
        system_prompt = """You are an expert HR assistant AI. Your task is to analyze a user's query and a list of candidate profiles to provide a professional, concise, and helpful hiring recommendation.
        - Start with a direct 1-2 sentence summary of the findings.
        - Identify the single **Top Recommendation** and explain *why* they are the best fit in 2-3 bullet points.
        - List any **Other Suitable Candidates** with a brief one-sentence justification for each.
        - If there are no perfect matches but some are close, state that and explain the trade-offs.
        - Maintain a professional and helpful tone. Use Markdown for formatting (bolding, lists).
        """
        
        user_prompt = f"""
        <query>
        {query}
        </query>
        
        <candidate_profiles>
        {context_str}
        </candidate_profiles>
        
        Please provide your hiring recommendation based on the query and profiles provided.
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant", # Use the current fast model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5, # Slightly lower temp for more factual, less creative responses
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error communicating with the LLM: {e}")
            # Fallback to the larger model if the first one fails for any reason (e.g., rate limits)
            try:
                response = self.llm_client.chat.completions.create(
                    model="llama-3.1-70b-versatile", # Use the current powerful model as a fallback
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.6,
                    max_tokens=1024,
                )
                return response.choices[0].message.content
            except Exception as e2:
                st.error(f"Fallback LLM also failed: {e2}")
                return "Sorry, I encountered an error while generating the recommendation. Please try again."

@st.cache_resource
def load_rag_system():
    """Caches the RAG system to avoid reloading models on every interaction."""
    try:
        return RAGSystem(api_key=st.secrets["GROQ_API_KEY"])
    except (ValueError, KeyError) as e:
        st.error(f"Initialization failed: {e}. Please ensure GROQ_API_KEY is set in your Streamlit secrets.")
        return None

# --- 4. UI Helper Functions ---
def display_employee(emp: Employee):
    """Renders a single employee's details in a formatted card."""
    availability_icon = "✅" if emp.availability == "available" else "⏳"
    availability_text = "Available" if emp.availability == "available" else emp.availability.replace("_", " ").title()
    
    st.markdown(f"""
    <div class="employee-card">
        <h3>👤 {emp.name}</h3>
        <p><b>Experience:</b> {emp.experience_years} years</p>
        <p><b>Status:</b> {availability_icon} {availability_text}</p>
        <p><b>Skills:</b> {", ".join(emp.skills)}</p>
        <p><b>Projects:</b> {", ".join(emp.projects)}</p>
        {f"<p><b>Notes:</b> {emp.notes}</p>" if emp.notes else ""}
    </div>
    """, unsafe_allow_html=True)

# --- 5. Main Streamlit App Flow ---
def main():
    st.markdown('<h1 class="title-text">Talent Finder AI ✨</h1>', unsafe_allow_html=True)

    rag_system = load_rag_system()
    if not rag_system:
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "employees" in message and message["employees"]:
                with st.expander("👥 Recommended Candidates", expanded=False):
                    for emp in message["employees"]:
                        display_employee(emp)

    # Handle new user input
    if user_query := st.chat_input("Find talent (e.g., 'Python developer with 5+ years experience in machine learning')"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching for candidates..."):
                employees = rag_system.search(user_query)
            
            if employees:
                with st.spinner("Generating recommendation..."):
                    llm_answer = rag_system.ask_llm(user_query, employees)
                
                st.markdown(llm_answer)
                with st.expander("👥 Recommended Candidates", expanded=True):
                    for emp in employees:
                        display_employee(emp)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": llm_answer, 
                    "employees": employees
                })
            else:
                no_results_message = "I couldn't find any candidates matching those specific criteria. You could try broadening your search, for example, by removing a skill or reducing the years of experience."
                st.write(no_results_message)
                st.session_state.messages.append({"role": "assistant", "content": no_results_message, "employees": []})

if __name__ == "__main__":
    main()
