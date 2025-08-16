# app.py

import os
import json
import re
import time
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum, auto

import numpy as np
import faiss
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# --- 1. Page Configuration & UI Styling ---
st.set_page_config(page_title="Talent Finder AI", page_icon="✨", layout="wide")

# (CSS is unchanged and has been collapsed for brevity)
st.markdown("""<style> .stApp { background-color: #0d1117; } @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } } @keyframes pulse { 0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(0, 184, 255, 0.7); } 70% { transform: scale(1.02); box-shadow: 0 0 10px 15px rgba(0, 184, 255, 0); } 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(0, 184, 255, 0); } } .title-text { font-size: 3rem; font-weight: 700; text-align: center; margin-bottom: 1rem; background: -webkit-linear-gradient(45deg, #00FFA3, #00B8FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: fadeIn 1s ease-out forwards; } .loader-container { text-align: center; padding: 20px; font-size: 1.1rem; color: #8b949e; animation: fadeIn 0.5s ease-out forwards; } .loader-container .robot-icon { font-size: 2.5rem; display: block; margin-bottom: 10px; animation: pulse 2s infinite; } .stButton>button { border: 1px solid #2d333b; border-radius: 8px; background-color: #161b22; color: #c9d1d9; transition: all 0.3s ease; animation: fadeIn 0.5s ease-out forwards; } .stButton>button:hover { border-color: #00B8FF; color: #00B8FF; transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0, 184, 255, 0.2); } .employee-card { border: 1px solid #2d333b; border-radius: 12px; padding: 20px; background-color: #161b22; box-shadow: 0 4px 8px rgba(0,0,0,0.2); transition: transform 0.3s ease, box-shadow 0.3s ease; animation: fadeIn 0.5s ease-out forwards; height: 100%; } .employee-card:hover { transform: translateY(-5px); box-shadow: 0 8px 20px rgba(0, 184, 255, 0.25); border-color: #00B8FF; } .employee-card h3 { color: #00B8FF; margin-top: 0; } .employee-card p { color: #c9d1d9; font-size: 0.95rem; } .employee-card summary { color: #8b949e; cursor: pointer; } .employee-card .icon { font-size: 1.1em; margin-right: 5px; } </style>""", unsafe_allow_html=True)


# --- 2. Data and Models ---
EMPLOYEE_DATA = { "employees": [ { "id": 1, "name": "Alice Johnson", "skills": ["Python", "React", "AWS", "Node.js"], "experience_years": 5, "projects": ["E-commerce Platform Migration", "Healthcare Dashboard UI"], "availability": "available" }, { "id": 2, "name": "Dr. Sarah Chen", "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "Medical Data Processing"], "experience_years": 6, "projects": ["Medical Diagnosis Platform (Computer Vision)", "Genomic Data Analysis Pipeline"], "availability": "available", "notes": "Published 3 papers on healthcare AI." }, { "id": 3, "name": "Michael Rodriguez", "skills": ["Python", "Machine Learning", "scikit-learn", "pandas", "HIPAA Compliance"], "experience_years": 4, "projects": ["Patient Risk Prediction System", "EHR Anonymization Tool"], "availability": "on project until 2025-10-15" }, { "id": 4, "name": "David Smith", "skills": ["Java", "Spring Boot", "Microservices", "Kafka", "PostgreSQL"], "experience_years": 8, "projects": ["Financial Trading System Backend", "Banking API Gateway"], "availability": "available" }, { "id": 5, "name": "Emily White", "skills": ["React Native", "JavaScript", "TypeScript", "Firebase", "GraphQL"], "experience_years": 3, "projects": ["Mobile Banking App", "Social Media Content App"], "availability": "available" }, { "id": 6, "name": "Chris Green", "skills": ["DevOps", "Kubernetes", "Docker", "AWS", "Terraform", "CI/CD"], "experience_years": 7, "projects": ["Cloud Infrastructure Automation", "CI/CD Pipeline Optimization"], "availability": "available" }, { "id": 7, "name": "Priya Patel", "skills": ["Data Science", "R", "SQL", "Tableau", "PowerBI"], "experience_years": 4, "projects": ["Customer Churn Analysis Dashboard", "Marketing Campaign ROI Prediction"], "availability": "available" }, { "id": 8, "name": "Tom Clark", "skills": ["Go", "gRPC", "Prometheus", "System Design"], "experience_years": 6, "projects": ["High-performance Logging Service", "Real-time Bidding System"], "availability": "on project until 2025-11-01" }, { "id": 9, "name": "Laura Martinez", "skills": ["UX/UI Design", "Figma", "Sketch", "User Research"], "experience_years": 5, "projects": ["Redesign of an e-learning platform", "User journey mapping for a fintech app"], "availability": "available" }, { "id": 10, "name": "James Wilson", "skills": ["Python", "Django", "PostgreSQL", "Celery", "Redis"], "experience_years": 9, "projects": ["Scalable Web App for Logistics", "Content Management System"], "availability": "available" }, { "id": 11, "name": "Zoe Brown", "skills": ["Cybersecurity", "Penetration Testing", "Metasploit", "Wireshark"], "experience_years": 5, "projects": ["Security Audit for a financial institution", "Network Vulnerability Assessment"], "availability": "on project until 2025-09-30" }, { "id": 12, "name": "Ethan Hunt", "skills": ["React", "Next.js", "Vercel", "TailwindCSS"], "experience_years": 3, "projects": ["Corporate Website Overhaul", "Server-side Rendered Marketing Site"], "availability": "available" }, { "id": 13, "name": "Grace Lee", "skills": ["Project Management", "Agile", "Scrum", "Jira"], "experience_years": 10, "projects": ["Led development of 'E-commerce Platform Migration'", "Coordinated 'Mobile Banking App' launch"], "availability": "available" }, { "id": 14, "name": "Ben Carter", "skills": ["AWS", "Docker", "Python", "Bash Scripting", "Ansible"], "experience_years": 4, "projects": ["Automated cloud deployment scripts", "Containerization of legacy Java application"], "availability": "available" }, { "id": 15, "name": "Olivia Garcia", "skills": ["Java", "Android", "Kotlin", "Jetpack Compose"], "experience_years": 4, "projects": ["Android App for a restaurant chain", "Fitness Tracking Mobile App"], "availability": "available" } ] }

class Employee(BaseModel):
    id: int
    name: str
    skills: List[str]
    experience_years: int
    projects: List[str]
    availability: str
    notes: Optional[str] = None

# ✨ NEW: Enum to represent the user's intent
class UserIntent(Enum):
    FIND_PEOPLE = auto()
    LIST_ALL = auto()
    CHITCHAT = auto()

# --- 3. RAG System (Backend Logic) ---
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
        self.all_skills = {skill.lower() for emp in self.employees for skill in emp['skills']}

    def _create_documents(self) -> List[str]:
        return [f"Name: {e['name']}. Skills: {', '.join(e['skills'])}. Experience: {e['experience_years']} years. Projects: {', '.join(e['projects'])}. Notes: {e.get('notes', 'N/A')}" for e in self.employees]

    def _create_faiss_index(self):
        embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=False)
        index = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings.shape[1]))
        ids = np.array([emp['id'] for emp in self.employees])
        index.add_with_ids(embeddings, ids)
        return index

    def _call_llm(self, user_prompt: str, system_prompt: str, model: str = "llama3-8b-8192", temperature: float = 0.5, is_json: bool = False) -> str:
        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"} if is_json else None,
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"LLM API Error: {e}")
            return '{"error": "Failed to get a response from the AI."}' if is_json else "I'm sorry, I encountered an error while generating a response."

    # ✨ NEW: Intelligent query analysis using an LLM
    def analyze_query(self, query: str) -> Tuple[UserIntent, str]:
        """
        Classifies the user's intent and expands the query if it's for finding people.
        """
        system_prompt = f"""
        You are an intelligent query analyzer for an HR system. Your task is to determine the user's intent and, if they are searching for people, expand their query to be more effective for semantic search.

        Available intents are:
        - "FIND_PEOPLE": The user is looking for candidates with specific skills, experience, or project history.
        - "LIST_ALL": The user wants a complete list of all employees. Keywords: "list all", "show all", "show everyone".
        - "CHITCHAT": The user is having a general conversation, asking about you, or greeting you. Keywords: "who are you", "who made you", "hello", "what can you do".

        Based on the user query, return a JSON object with two keys:
        1. "intent": One of the three intents listed above.
        2. "expanded_query":
           - If the intent is "FIND_PEOPLE", rephrase the user's query into a detailed description of the ideal candidate. Example: "python dev with 5 years" becomes "A software developer with at least 5 years of experience specializing in Python programming.".
           - If the intent is "LIST_ALL" or "CHITCHAT", return the original user query verbatim.

        Respond ONLY with the JSON object.
        """
        response_str = self._call_llm(query, system_prompt, is_json=True, temperature=0.1)
        try:
            result = json.loads(response_str)
            intent_str = result.get("intent", "CHITCHAT")
            expanded_query = result.get("expanded_query", query)
            
            intent_map = {
                "FIND_PEOPLE": UserIntent.FIND_PEOPLE,
                "LIST_ALL": UserIntent.LIST_ALL,
                "CHITCHAT": UserIntent.CHITCHAT,
            }
            return intent_map.get(intent_str, UserIntent.CHITCHAT), expanded_query
        except (json.JSONDecodeError, KeyError):
            # If LLM fails to produce valid JSON, default to a safe option
            return UserIntent.FIND_PEOPLE, query

    def _parse_and_get_filtered_ids(self, query: str) -> Set[int]:
        """Performs hard filtering based on explicit criteria in the query."""
        query_lower = query.lower()
        candidate_ids = set(self.employee_map.keys())
        
        # Filter by experience
        exp_match = re.search(r'(\d+)\+?\s*years', query_lower)
        if exp_match:
            min_exp = int(exp_match.group(1))
            candidate_ids.intersection_update({emp['id'] for emp in self.employees if emp['experience_years'] >= min_exp})
            
        # Filter by skills
        required_skills = {skill for skill in self.all_skills if skill in query_lower}
        if required_skills:
            candidate_ids.intersection_update({
                emp['id'] for emp in self.employees 
                if all(req_skill in [s.lower() for s in emp['skills']] for req_skill in required_skills)
            })
        return candidate_ids

    def search(self, original_query: str, expanded_query: str) -> List[Employee]:
        """
        Performs a dynamic hybrid search using both metadata filtering and semantic search.
        """
        # Use the original query for hard filters to catch specific keywords like "5+ years"
        pre_filtered_ids = self._parse_and_get_filtered_ids(original_query)

        if not pre_filtered_ids:
            return []

        # Use the expanded query for a more nuanced semantic search
        query_embedding = self.embedding_model.encode([expanded_query])
        
        # Search across all candidates to get a semantic ranking
        distances, semantic_ids_list = self.index.search(query_embedding, k=len(self.employees))

        # Combine results: Rank by semantic similarity, but only include those who pass the hard filters
        final_candidates = []
        seen_ids = set()
        for eid in semantic_ids_list[0]:
            if eid != -1 and eid in pre_filtered_ids and eid not in seen_ids:
                final_candidates.append(Employee(**self.employee_map[eid]))
                seen_ids.add(eid)
                
        return final_candidates

    # ✨ UPDATED: More detailed and professional system prompts
    def generate_hr_response(self, query: str, context_employees: List[Employee]) -> str:
        system_prompt = """
        You are an expert HR Talent Acquisition Partner AI. Your tone is professional, insightful, and concise.

        You will be given a user's query and a list of JSON profiles for suitable candidates retrieved from the database.
        
        Your task is to generate a helpful and well-structured response that includes:
        1.  **Summary**: Briefly acknowledge the user's request and state that you have found some suitable candidates.
        2.  **Top Recommendations**: Highlight 1-2 top candidates who are the strongest match. Briefly explain *why* they are a good fit, referencing their specific skills or project experience.
        3.  **Conclusion**: End with a confident and helpful closing statement, perhaps suggesting next steps.

        **Do not just list the employees.** Provide a valuable, synthesized analysis. Do not mention that the data was provided to you in JSON format.
        """
        context_str = "\n---\n".join([json.dumps(emp.model_dump()) for emp in context_employees])
        user_prompt = f"""User Query: "{query}"\n\nRetrieved Candidate Profiles:\n{context_str}\n\nBased on this, generate your expert recommendation."""
        return self._call_llm(user_prompt, system_prompt, temperature=0.7)

    def generate_general_response(self, query: str) -> str:
        system_prompt = """
        You are a friendly and helpful conversational AI assistant for a Talent Finder application.
        - If the user greets you, greet them back warmly.
        - If the user asks who you are or what you do, explain that you are an intelligent HR assistant designed to help find talent from an internal employee database.
        - If the user asks who made you, state that you were created by Vicky Mahato as an advanced AI project.
        - For any other general chit-chat, be helpful and conversational.
        - If you cannot find a candidate, state it clearly and politely suggest broadening the search criteria.
        """
        return self._call_llm(query, system_prompt, temperature=0.7)

@st.cache_resource
def load_rag_system():
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it to run the application.")
            return None
        return RAGSystem(api_key=api_key)
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return None

# --- 4. UI Helper Functions (Unchanged) ---
def stream_response(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.02)

def display_employee_card(card_data: dict, container):
    with container:
        st.markdown(f"""<div class="employee-card"><h3><span class="icon">👤</span>{card_data['name']}</h3><p><span class="icon">📅</span><b>Experience:</b> {card_data['experience_years']} years</p><p><span class="icon">📌</span><b>Status:</b> {'✅ Available' if card_data['availability'].lower() == 'available' else '⏳ ' + card_data['availability']} </p><details><summary><b>View Details</b></summary><p><span class="icon">🛠️</span><b>Skills:</b> {', '.join(card_data['skills'])}</p><p><span class="icon">🚀</span><b>Projects:</b></p><ul>{''.join(f"<li>{proj}</li>" for proj in card_data['projects'])}</ul></details>{f"<p><span class='icon'>📝</span><b>Notes:</b> {card_data.get('notes')}</p>" if card_data.get('notes') else ""}</div>""", unsafe_allow_html=True)

def show_thinking_animation(step_text: str):
    placeholder = st.empty()
    placeholder.markdown(f"""<div class="loader-container"><div class="robot-icon">🤖</div><div>{step_text}</div></div>""", unsafe_allow_html=True)
    time.sleep(1) # Simulate work
    placeholder.empty()

def handle_prompt_click(prompt_text):
    st.session_state.clicked_prompt = prompt_text

def display_cards_grid(cards):
    if not cards:
        return
    
    num_cards = len(cards)
    cols_per_row = 3
    num_rows = (num_cards + cols_per_row - 1) // cols_per_row
    
    for i in range(num_rows):
        cols = st.columns(cols_per_row)
        row_cards = cards[i * cols_per_row:(i + 1) * cols_per_row]
        for j, card in enumerate(row_cards):
            display_employee_card(card, cols[j])


# --- 5. Main Application ---
# Sidebar (Unchanged)
with st.sidebar:
    st.markdown("### 🤖 Talent Finder AI")
    st.markdown("This AI chatbot uses a hybrid search system to find the right talent.")
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.clicked_prompt = None
        st.rerun()

st.markdown('<h1 class="title-text">Talent Finder AI ✨</h1>', unsafe_allow_html=True)
rag_system = load_rag_system()

if rag_system:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "clicked_prompt" not in st.session_state:
        st.session_state.clicked_prompt = None

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("cards"):
                expander_title = "👥 View All Employee Profiles" if message.get("is_list_all") else "👥 View Recommended Candidate Profiles"
                with st.expander(expander_title, expanded=False):
                    display_cards_grid(message["cards"])

    # Show welcome message and example prompts if chat is empty
    if not st.session_state.messages:
        st.markdown("<div style='text-align: center;'><h2>Welcome!</h2><p>Ask me to find talent, or try an example:</p></div>", unsafe_allow_html=True)
        cols = st.columns(4)
        prompts = ["List all employees", "Python devs with 5+ years", "Who knows AWS and Docker?", "Who made you?"]
        buttons = [
            cols[0].button(prompts[0], use_container_width=True, on_click=handle_prompt_click, args=[prompts[0]]),
            cols[1].button(prompts[1], use_container_width=True, on_click=handle_prompt_click, args=[prompts[1]]),
            cols[2].button(prompts[2], use_container_width=True, on_click=handle_prompt_click, args=[prompts[2]]),
            cols[3].button(prompts[3], use_container_width=True, on_click=handle_prompt_click, args=[prompts[3]]),
        ]

    # Chat input
    _, input_col, _ = st.columns([1, 3, 1])
    with input_col:
        prompt = st.chat_input("Find an employee...") or st.session_state.clicked_prompt

    if prompt:
        st.session_state.clicked_prompt = None
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # ✨ NEW: Centralized, intelligent response generation
            show_thinking_animation("🤔 Analyzing your request...")
            intent, expanded_query = rag_system.analyze_query(prompt)

            answer = ""
            cards_to_show = []
            is_list_all = False

            match intent:
                case UserIntent.FIND_PEOPLE:
                    show_thinking_animation("🧠 Searching for candidates...")
                    retrieved_employees = rag_system.search(original_query=prompt, expanded_query=expanded_query)
                    
                    if retrieved_employees:
                        answer = rag_system.generate_hr_response(prompt, retrieved_employees)
                        cards_to_show = [emp.model_dump() for emp in retrieved_employees]
                    else:
                        answer = rag_system.generate_general_response(f"I couldn't find anyone who matches the query: '{prompt}'. Could you try broadening your search?")
                    
                case UserIntent.LIST_ALL:
                    answer = "Here is a complete list of all employees in our talent pool:"
                    cards_to_show = rag_system.employees
                    is_list_all = True

                case UserIntent.CHITCHAT:
                    answer = rag_system.generate_general_response(prompt)

            # Stream the text response and display cards
            st.write_stream(stream_response(answer))
            if cards_to_show:
                expander_title = "👥 View All Employee Profiles" if is_list_all else "👥 View Recommended Candidate Profiles"
                with st.expander(expander_title, expanded=True):
                   display_cards_grid(cards_to_show)
            
            # Save the complete message to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "cards": cards_to_show,
                "is_list_all": is_list_all
            })
            st.rerun()

else:
    st.warning("Application could not start. Please verify your API key in Streamlit secrets.")
