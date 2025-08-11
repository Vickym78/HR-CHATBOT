# --- Optimized RAG Search System with Better Prompt Handling ---
import os
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
from difflib import get_close_matches

st.set_page_config(page_title="Talent Finder AI", page_icon="✨", layout="wide")

# -------------------- CSS Styling --------------------
st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    .title-text { font-size: 3rem; font-weight: 700; text-align: center;
        margin-bottom: 1rem; background: -webkit-linear-gradient(45deg, #00FFA3, #00B8FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .employee-card { border: 1px solid #2d333b; border-radius: 12px;
        padding: 20px; background-color: #161b22; transition: transform 0.3s ease; }
    .employee-card:hover { transform: translateY(-5px); border-color: #00B8FF; }
    .employee-card h3 { color: #00B8FF; }
    .employee-card p { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

# -------------------- Employee Data --------------------
EMPLOYEE_DATA = {...}  # Keep your employee JSON as is

class Employee(BaseModel):
    id: int
    name: str
    skills: List[str]
    experience_years: int
    projects: List[str]
    availability: str
    notes: Optional[str] = None

# -------------------- Synonym & Keyword Mappings --------------------
SKILL_SYNONYMS = {
    "frontend": ["react", "javascript", "html", "css"],
    "backend": ["node.js", "java", "spring boot", "django"],
    "ml": ["machine learning", "tensorflow", "pytorch"],
    "devops": ["docker", "kubernetes", "aws", "ci/cd"]
}

# -------------------- RAG System --------------------
class RAGSystem:
    def __init__(self, api_key: str):
        self.employees = EMPLOYEE_DATA['employees']
        self.employee_map = {emp['id']: emp for emp in self.employees}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = self._create_documents()
        self.index = self._create_faiss_index()
        self.llm_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        self.all_skills = {skill.lower() for emp in self.employees for skill in emp['skills']}

    def _create_documents(self) -> List[str]:
        return [f"{e['name']} | Skills: {', '.join(e['skills'])} | Exp: {e['experience_years']} yrs"
                for e in self.employees]

    def _create_faiss_index(self):
        embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=False)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
        ids = np.array([emp['id'] for emp in self.employees])
        index.add_with_ids(embeddings, ids)
        return index

    def _expand_query(self, query: str) -> str:
        query_lower = query.lower()
        for keyword, synonyms in SKILL_SYNONYMS.items():
            if keyword in query_lower:
                query_lower += " " + " ".join(synonyms)
        return query_lower

    def _parse_filters(self, query: str) -> Set[int]:
        query_lower = query.lower()
        candidate_ids = set(self.employee_map.keys())

        exp_match = re.search(r'(\d+)\+?\s*years', query_lower)
        if exp_match:
            min_exp = int(exp_match.group(1))
            candidate_ids &= {emp['id'] for emp in self.employees if emp['experience_years'] >= min_exp}

        skills = {skill for skill in self.all_skills if skill in query_lower}
        if skills:
            candidate_ids &= {emp['id'] for emp in self.employees
                              if all(s in [sk.lower() for sk in emp['skills']] for s in skills)}
        return candidate_ids

    def search(self, query: str, top_k: int = 10):
        query = self._expand_query(query)
        filters = self._parse_filters(query)

        if not filters:
            return []

        query_vec = self.embedding_model.encode([query])
        faiss.normalize_L2(query_vec)
        distances, ids = self.index.search(query_vec, k=top_k)

        results = [self.employee_map[eid] for eid in ids[0] if eid in filters]
        return [Employee(**emp) for emp in results]

@st.cache_resource
def load_rag():
    return RAGSystem(api_key=st.secrets["GROQ_API_KEY"])

# -------------------- Display Function --------------------
def display_cards(cards):
    if not cards:
        st.warning("No matching candidates found.")
        return
    rows = [cards[i:i+3] for i in range(0, len(cards), 3)]
    for row in rows:
        cols = st.columns(len(row))
        for col, card in zip(cols, row):
            with col:
                st.markdown(f"""
                <div class="employee-card">
                    <h3>{card.name}</h3>
                    <p>Experience: {card.experience_years} years</p>
                    <p>Skills: {', '.join(card.skills)}</p>
                    <p>Status: {card.availability}</p>
                </div>
                """, unsafe_allow_html=True)

# -------------------- Main UI --------------------
st.markdown('<h1 class="title-text">Talent Finder AI ✨</h1>', unsafe_allow_html=True)

rag = load_rag()
query = st.chat_input("Search for a candidate...")

if query:
    results = rag.search(query)
    display_cards(results)
