# app.py

import streamlit as st
import requests
import time

# --- Page Configuration ---
st.set_page_config(page_title="Query Chatbot 🤖", page_icon="🤖", layout="wide")

# --- Helper Functions ---
def local_css():
    """Injects custom CSS for styling UI elements."""
    st.markdown("""
    <style>
        .employee-card {
            border-radius: 15px; border: 1px solid #e0e0e0; padding: 20px;
            margin: 10px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease-in-out; background-color: #ffffff;
        }
        .employee-card:hover {
            box-shadow: 0 8px 16px rgba(0,0,0,0.2); transform: translateY(-5px);
            border-color: #4A90E2;
        }
    </style>
    """, unsafe_allow_html=True)

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
            if card_data['availability'] == 'available':
                st.success("**Status:** Available")
            else:
                st.warning(f"**Status:** {card_data['availability']}")
        with st.expander("🛠️ View Skills & Projects"):
            st.write(f"**Skills:** `{', '.join(card_data['skills'])}`")
            st.write("**Past Projects:**")
            for project in card_data['projects']:
                st.markdown(f"- *{project}*")

# --- Setup ---
local_css()
API_BASE_URL = "http://127.0.0.1:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("HR Assistant Pro 🤖")
    st.info("Find the right talent instantly.")
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Title ---
st.title("HR Resource Query Chatbot")

# --- Show Chat History ---
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "cards" in message and message["cards"]:
            for card in message["cards"]:
                display_employee_card(card)

# --- Chat Input & Example Prompts ---
st.markdown("---")
cols = st.columns([8, 1, 1, 1]) # Make input wider
prompt = cols[0].chat_input("Ask me to find an employee...")

with cols[1]:
    if st.button("Py", help="Find Python developers with 3+ years experience"):
        prompt = "Find Python developers with 3+ years experience"
with cols[2]:
    if st.button("RN", help="Suggest people for a React Native project"):
        prompt = "Suggest people for a React Native project"
with cols[3]:
    if st.button("DO", help="Find developers who know both AWS and Docker"):
        prompt = "Find developers who know both AWS and Docker"

# --- Process New Prompt ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤖 Searching talent..."):
            try:
                response = requests.post(f"{API_BASE_URL}/chat", json={"query": prompt}, timeout=60)
                response.raise_for_status()
                response_data = response.json()

                full_response = response_data.get("answer", "No matching employees found.")
                retrieved_employees = response_data.get("retrieved_employees", [])
                
                # Stream the text response
                st.write_stream(stream_response(full_response))
                
                # Display employee cards after the text
                if retrieved_employees:
                    for card in retrieved_employees:
                        display_employee_card(card)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "cards": retrieved_employees
                })

            except requests.exceptions.RequestException as e:
                error_msg = f"❌ Could not connect to backend API. Please ensure it's running. Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "cards": []})
            except Exception as e:
                error_msg = f"⚠️ An unexpected error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "cards": []})
    st.rerun()
