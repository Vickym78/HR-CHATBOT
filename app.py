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
st.set_page_config(page_title="Talent Finder AI", page_icon="✨", layout="wide")

# Custom CSS for a sleek, modern, and animated interface
st.markdown("""
<style>
    /* General Styles */
    .stApp {
        background-color: #0d1117;
    }

    /* Keyframe Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(0, 184, 255, 0.7); }
        70% { transform: scale(1.02); box-shadow: 0 0 10px 15px rgba(0, 184, 255, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(0, 184, 255, 0); }
    }

    /* Title Animation */
    .title-text {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        background: -webkit-linear-gradient(45deg, #00FFA3, #00B8FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 1s ease-out forwards;
    }

    /* Animated "Thinking" Loader */
    .loader-container {
        text-align: center;
        padding: 20px;
        font-size: 1.1rem;
        color: #8b949e;
        animation: fadeIn 0.5s ease-out forwards;
    }
    .loader-container .robot-icon {
        font-size: 2.5rem;
        display: block;
        margin-bottom: 10px;
        animation: pulse 2s infinite;
    }
    
    /* Example Prompt Buttons */
    .stButton>button {
        border: 1px solid #2d333b;
        border-radius: 8px;
        background-color: #161b22;
        color: #c9d1d9;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out forwards;
    }
    .stButton>button:hover {
        border-color: #00B8FF;
        color: #00B8FF;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 184, 255, 0.2);
    }
    
    /* Employee Card Styling with Animation */
    .employee-card {
        border: 1px solid #2d333b;
        border-radius: 12px;
        padding: 20px;
        background-color: #161b22;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.5s ease-out forwards;
        height: 100%; /* Ensure cards in a row have same height */
    }
    .employee-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 184, 255, 0.25);
        border-color: #00B8FF;
    }
    .employee-card h3 { color: #00B8FF; margin-top: 0; }
    .employee-card p { color: #c9d1d9; font-size: 0.95rem; }
    .employee-card summary { color: #8b949e; cursor: pointer; }
    .employee-card .icon { font-size: 1.1em; margin-right: 5px; }
</style>
""", unsafe_allow_html=True)


# --- 2. Data and Models ---
EMPLOYEE_DATA = {
  "employees": [
    { "id": 1, "name": "Alice Johnson", "skills": ["Python", "React", "AWS", "Node.js"], "experience_years": 5, "projects": ["E-commerce Platform Migration", "Healthcare Dashboard UI"], "availability": "available" },
    { "id": 2, "name": "Dr. Sarah Chen", "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "Medical Data Processing"], "experience_years": 6, "projects": ["Medical Diagnosis Platform (Computer Vision)", "Genomic Data Analysis Pipeline"], "availability": "available", "notes": "Published 3 papers on healthcare AI." },
    { "id": 3, "name": "Michael Rodriguez", "skills": ["Python", "Machine Learning", "scikit-learn", "pandas", "HIPAA Compliance"], "experience_years": 4, "projects": ["Patient Risk Prediction System", "EHR Anonymization Tool"], "availability": "on project until 2025-10-15" },
    { "id": 4, "name": "David Smith", "skills": ["Java", "Spring Boot", "Microservices", "Kafka", "PostgreSQL"], "experience_years": 8, "projects": ["Financial Trading System Backend", "Banking API Gateway"], "availability": "available" },
    { "id": 5, "name": "Emily White", "skills": ["React Native", "JavaScript", "TypeScript", "Firebase", "GraphQL"], "experience_years": 3, "projects": ["Mobile Banking App", "Social Media Content App"], "availability": "available" },
    { "id": 6, "name": "Chris Green", "skills": ["DevOps", "Kubernetes", "Docker", "AWS", "Terraform", "CI/CD"], "experience_years": 7, "projects": ["Cloud Infrastructure Automation", "CI/CD Pipeline Optimization"], "availability": "available" },
    { "id": 7, "name": "Priya Patel", "skills": ["Data Science", "R", "SQL", "Tableau", "PowerBI"], "experience_years": 4, "projects": ["Customer Churn Analysis Dashboard", "Marketing Campaign ROI Prediction"], "availability": "available" },
    { "id": 8, "name": "Tom Clark", "skills": ["Go", "gRPC", "Prometheus", "System Design"], "experience_years": 6, "projects": ["High-performance Logging Service", "Real-time Bidding System"], "availability": "on project until 2025-11-01" },
    { "id": 9, "name": "Laura Martinez", "skills": ["UX/UI Design", "Figma", "Sketch", "User Research"], "experience_years": 5, "projects": ["Redesign of an e-learning platform", "User journey mapping for a fintech app"], "availability": "available" },
    { "id": 10, "name": "James Wilson", "skills": ["Python", "Django", "PostgreSQL", "Celery", "Redis"], "experience_years": 9, "projects": ["Scalable Web App for Logistics", "Content Management System"], "availability": "available" },
    { "id": 11, "name": "Zoe Brown", "skills": ["Cybersecurity", "Penetration Testing", "Metasploit", "Wireshark"], "experience_years": 5, "projects": ["Security Audit for a financial institution", "Network Vulnerability Assessment"], "availability": "on project until 2025-09-30" },
    { "id": 12, "name": "Ethan Hunt", "skills": ["React", "Next.js", "Vercel", "TailwindCSS"], "experience_years": 3, "projects": ["Corporate Website Overhaul", "Server-side Rendered Marketing Site"], "availability": "available" },
    { "id": 13, "name": "Grace Lee", "skills": ["Project Management", "Agile", "Scrum", "Jira"], "experience_years": 10, "projects": ["Led development of 'E-commerce Platform Migration'", "Coordinated 'Mobile Banking App' launch"], "availability": "available" },
    { "id": 14, "name": "Ben Carter", "skills": ["AWS", "Docker", "Python", "Bash Scripting", "Ansible"], "experience_years": 4, "projects": ["Automated cloud deployment scripts", "Containerization of legacy Java application"], "availability": "available" },
    { "id": 15, "name": "Olivia Garcia", "skills": ["Java", "Android", "Kotlin", "Jetpack Compose"], "experience_years": 4, "projects": ["Android App for a restaurant chain", "Fitness Tracking Mobile App"], "availability": "available" }
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
        return [f"Name: {e['name']}. Skills: {', '.join(e['skills'])}. Experience: {e['experience_years']} years. Projects: {', '.join(e['projects'])}. Notes: {e.get('notes', 'N/A')}" for e in self.employees]

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
            response = self.llm_client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.7)
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"LLM API Error: {e}")
            return "I'm sorry, I encountered an error while generating a response."

    def generate_hr_response(self, query: str, context_employees: List[Employee]) -> str:
        system_prompt = """
        You are an expert HR Talent Acquisition Partner. Your goal is to provide a detailed, persuasive, and personalized recommendation based on the user's request and the provided candidate data.

        Follow these rules strictly:
        1.  **Acknowledge the Query:** Start with a brief introductory sentence that acknowledges the user's request.
        2.  **Detailed Candidate Analysis:**
            - Present each candidate in a separate, well-defined section using their name as a sub-header.
            - For each candidate, **do not just list their skills or projects.** You MUST synthesize this information.
            - **Crucially, explain *why* they are a perfect fit by explicitly connecting their specific skills and past project experience to the keywords and intent of the user's query.** For example, if the query is about "healthcare ML," highlight their project named "Medical Diagnosis Platform" and explain its relevance.
        3.  **Persuasive Tone:** Use confident and professional language to build trust in your recommendations.
        4.  **Proactive Closing:** Conclude your entire response with a helpful, proactive statement, suggesting next steps (e.g., "Would you like me to provide more details about their specific projects?" or "I can check their calendars for a meeting.").
        5.  **Formatting:** Use Markdown extensively (bolding, italics, lists) to make the response highly readable and professional.
        """
        context_str = "\n---\n".join([json.dumps(emp.model_dump()) for emp in context_employees])
        user_prompt = f"""
        User Query: "{query}"

        Retrieved Candidate Profiles:
        {context_str}

        Based on the provided user query and candidate profiles, please generate your expert recommendation following all the rules I've given you.
        """
        return self._call_llm(user_prompt, system_prompt)

    def generate_general_response(self, query: str) -> str:
        system_prompt = "You are a friendly and helpful conversational AI assistant. Use Markdown for all formatting."
        return self._call_llm(query, system_prompt)

@st.cache_resource
def load_rag_system():
    try:
        return RAGSystem(api_key=st.secrets.get("GROQ_API_KEY"))
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return None

# --- 4. UI Helper Functions ---
def stream_response(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.02)

def display_employee_card(card_data: dict, container):
    with container:
        st.markdown(
            f"""
            <div class="employee-card">
                <h3><span class="icon">👤</span>{card_data['name']}</h3>
                <p><span class="icon">📅</span><b>Experience:</b> {card_data['experience_years']} years</p>
                <p><span class="icon">📌</span><b>Status:</b> {'✅ Available' if card_data['availability'].lower() == 'available' else '⏳ ' + card_data['availability']}</p>
                <details>
                    <summary><b>View Details</b></summary>
                    <p><span class="icon">🛠️</span><b>Skills:</b> {', '.join(card_data['skills'])}</p>
                    <p><span class="icon">🚀</span><b>Projects:</b></p>
                    <ul>{''.join(f"<li>{proj}</li>" for proj in card_data['projects'])}</ul>
                </details>
                {f"<p><span class='icon'>📝</span><b>Notes:</b> {card_data['notes']}</p>" if card_data.get('notes') else ""}
            </div>
            """,
            unsafe_allow_html=True
        )

def show_thinking_animation():
    thinking_steps = ["🔍 Searching database...", "🧠 Analyzing candidates...", "✍️ Composing response..."]
    placeholder = st.empty()
    for step in thinking_steps:
        placeholder.markdown(f"""
        <div class="loader-container">
            <div class="robot-icon">🤖</div>
            <div>{step}</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.7)
    placeholder.empty()

def handle_prompt_click(prompt_text):
    st.session_state.clicked_prompt = prompt_text

# --- 5. Main Application ---
with st.sidebar:
    st.header("About")
    st.markdown("This AI-powered chatbot helps HR teams find the right talent by answering natural language queries.")
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("cards"):
                with st.expander("👥 View Recommended Candidate Profiles", expanded=False):
                    num_cards = len(message["cards"])
                    cols = st.columns(num_cards if num_cards > 0 else 1)
                    for i, card in enumerate(message["cards"]):
                        display_employee_card(card, cols[i])

    if not st.session_state.messages:
        st.info("Ask me anything about our talent pool!")
        st.markdown("<div style='text-align: center; margin-bottom: 10px; animation: fadeIn 1s ease-out forwards;'>Or try one of these:</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        prompts = ["Find Python ML experts", "Who are you?", "Who is your developer?"]
        if cols[0].button(prompts[0], use_container_width=True, on_click=handle_prompt_click, args=[prompts[0]]): pass
        if cols[1].button(prompts[1], use_container_width=True, on_click=handle_prompt_click, args=[prompts[1]]): pass
        if cols[2].button(prompts[2], use_container_width=True, on_click=handle_prompt_click, args=[prompts[2]]): pass

    prompt = st.chat_input("e.g., 'Find developers with machine learning skills'") or st.session_state.clicked_prompt
    
    if prompt:
        st.session_state.clicked_prompt = None
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # --- IDENTITY & DEVELOPER CHECKS ---
            developer_keywords = ["who made you", "your developer", "created you", "invented you", "creator", "vicky"]
            identity_keywords = ["who are you", "what are you"]
            prompt_lower = prompt.lower()

            answer = ""
            cards_to_show = []

            if any(keyword in prompt_lower for keyword in developer_keywords):
                answer = "I was created by **Vicky Mahato**. He's a talented developer who built me to help HR teams find the best talent efficiently! 🚀"
                st.write_stream(stream_response(answer))

            elif any(keyword in prompt_lower for keyword in identity_keywords):
                answer = "I am an intelligent **HR Assistant Chatbot** 🤖, designed to help you find the best talent in our company. Ask me about our employees' skills or project experience!"
                st.write_stream(stream_response(answer))
            
            else:
                # --- Original RAG Logic ---
                show_thinking_animation()
                retrieved_employees, scores = rag_system.search(prompt)
                RELEVANCE_THRESHOLD = 1.0

                if retrieved_employees and scores[0][0] < RELEVANCE_THRESHOLD:
                    answer = rag_system.generate_hr_response(prompt, retrieved_employees)
                    cards_to_show = [emp.model_dump() for emp in retrieved_employees]
                else:
                    answer = rag_system.generate_general_response(prompt)
                    cards_to_show = []

                st.write_stream(stream_response(answer))
                if cards_to_show:
                    with st.expander("👥 View Recommended Candidate Profiles", expanded=True):
                        num_cards = len(cards_to_show)
                        cols = st.columns(num_cards if num_cards > 0 else 1)
                        for i, card in enumerate(cards_to_show):
                            display_employee_card(card, cols[i])
            
            # --- Save to history and rerun ---
            st.session_state.messages.append({"role": "assistant", "content": answer, "cards": cards_to_show})
            st.rerun()

else:
    st.warning("Application could not start. Please verify your API key in Streamlit secrets.")
