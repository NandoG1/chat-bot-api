import os
import streamlit as st
from dotenv import load_dotenv

from openai import OpenAI
import google.generativeai as genai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

print(OPENAI_API_KEY)
oai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="AI Chatbot", page_icon="-", layout="wide")

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not found in .env")
if not GEMINI_API_KEY:
    st.warning("GEMINI_API_KEY not found in .env")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "cost" not in st.session_state:
    st.session_state.cost = 0.0

st.sidebar.title("Settings")

provider = st.sidebar.selectbox("Choose Provider", ["OpenAI", "Gemini"])

default_model = "gpt-4o-mini" if provider == "OpenAI" else "gemini-2.5-flash"
model = st.sidebar.text_input("Model", default_model)

with st.sidebar.expander("Advanced Settings"):
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.95)
    top_k = st.slider("Top-k (Gemini only)", 1, 100, 40)
    max_tokens = st.slider("Max Tokens", 50, 2048, 512)

min_budget = 1.0
max_budget = 5.0
current_cost = st.session_state.cost
cost_color = "#008000" if current_cost <= max_budget else "#FF0000"

st.sidebar.markdown(f"""
<style>
.budget-box {{
    background-color: #f9f9f9;
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 10px;
    margin-bottom: 20px;
    font-family: 'Segoe UI', sans-serif;
}}
.budget-title {{
    color: black;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
}}
.budget-text {{
    color: black;
    font-size: 14px;
    margin: 6px 0;
}}
.budget-cost {{
    font-size: 15px;
    font-weight: 700;
    color: {cost_color};
    margin-top: 12px;
}}
</style>

<div class="budget-box">
    <div class="budget-title">Budget Control</div>
    <div class="budget-text">Min Budget: <b>${min_budget}</b></div>
    <div class="budget-text">Max Budget: <b>${max_budget}</b></div>
    <div class="budget-cost">Current Cost: ${current_cost:.4f}</div>
</div>
""", unsafe_allow_html=True)

def render_history():
    """Render past chat messages."""
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def to_gemini_history(messages):
    """Convert Streamlit chat history to Gemini format."""
    history = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        history.append({"role": role, "parts": [m["content"]]})
    return history

def estimate_cost_openai(tokens: int) -> float:
    # Example: $0.005 per 1k tokens => 0.000005 per token
    return tokens * 0.000005

def oai_chat(messages, user_prompt):
    if not oai_client:
        return "OPENAI_API_KEY not found."
    try:
        resp = oai_client.chat.completions.create(
            model=model,
            messages=messages + [{"role": "user", "content": user_prompt}],
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
        )
        reply = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        if usage and getattr(usage, "total_tokens", None) is not None:
            st.session_state.cost += estimate_cost_openai(usage.total_tokens)
        return reply
    except Exception as e:
        return f"OpenAI error: {e}"

def gemini_chat(messages, user_prompt):
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY not found."
    try:
        gmodel = genai.GenerativeModel(model)
        chat = gmodel.start_chat(history=to_gemini_history(messages))
        cfg = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "max_output_tokens": int(max_tokens),
        }
        resp = chat.send_message(user_prompt, generation_config=cfg)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"Gemini error: {e}"

def call_model(messages, user_prompt):
    """Route request to chosen provider."""
    if provider == "OpenAI":
        return oai_chat(messages, user_prompt)
    return gemini_chat(messages, user_prompt)

def summarize():
    if not st.session_state.messages:
        st.warning("No chat history yet.")
        return
    transcript = "\n".join(
        [f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}"
         for m in st.session_state.messages]
    )
    prompt = "Summarize the following conversation in 5 concise bullet points:\n\n" + transcript
    with st.spinner("Summarizing…"):
        summary = call_model([], prompt)
        with st.chat_message("assistant"):
            st.markdown("**Summary:**\n\n" + summary)
        st.session_state.messages.append(
            {"role": "assistant", "content": "Summary:\n" + summary}
        )

if st.sidebar.button("Summarize Chat"):
    summarize()

st.title("AI Chatbot")

render_history()

if user_input := st.chat_input("Type your message…"):
    # User message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = call_model(st.session_state.messages[:-1], user_input)
            st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})