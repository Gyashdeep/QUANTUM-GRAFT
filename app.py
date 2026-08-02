import asyncio
import os
from typing import List
from pydantic import BaseModel, Field
from groq import AsyncGroq
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="QUANTUM-GRAFT: A.I. Sovereign Swarm",
    page_icon="⚡",
    layout="wide",
)

# --- Cyberpunk / Industrial Terminal Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
        background-color: #050505;
        color: #00ff66;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #0a0a0a !important;
        color: #00ff66 !important;
        border: 1px solid #00ff66 !important;
        font-family: 'JetBrains Mono', monospace;
    }
    .stButton button {
        background-color: #00ff66 !important;
        color: #050505 !important;
        font-weight: bold;
        border-radius: 0px;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid #00ff66;
    }
    .stButton button:hover {
        background-color: #050505 !important;
        color: #00ff66 !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #080808;
        border-right: 1px solid #113311;
    }
</style>
""", unsafe_allow_html=True)

# --- Schemas & State ---
class AgentState(BaseModel):
    query: str
    retrieved_docs: List[str] = Field(default_factory=list)
    critique: str = ""
    final_output: str = ""
    iteration: int = 0

# --- Live Groq LPU Powered Agents (Switched to openai/gpt-oss-120b) ---

class RouterAgent:
    def __init__(self, client: AsyncGroq):
        self.client = client

    async def process(self, state: AgentState) -> str:
        response = await self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a specialized query optimization agent for high-precision hybrid retrieval systems. Output ONLY the optimized search query string without any conversational filler."},
                {"role": "user", "content": f"Optimize this query for vector search: {state.query}"}
            ],
            temperature=0.1,
            max_tokens=60
        )
        return response.choices[0].message.content.strip()

class RetrievalAgent:
    async def search(self, optimized_query: str) -> List[str]:
        await asyncio.sleep(0.1)
        return [
            "Context Chunk 1: AR-FT combines adaptive retrieval thresholds with domain-specific QLoRA weights to mitigate hallucination drift.",
            "Context Chunk 2: Multi-agent coordination graphs utilize state-machine loops to validate semantic alignment before final output synthesis."
        ]

class CriticAgent:
    def __init__(self, client: AsyncGroq):
        self.client = client

    async def evaluate(self, state: AgentState) -> str:
        context_preview = "\n".join(state.retrieved_docs)
        response = await self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a strict validation critic agent. Evaluate if the retrieved context adequately answers the initial user query. Respond with 'Approved: [Reason]' or 'Rejected: [Reason]'."},
                {"role": "user", "content": f"Query: {state.query}\n\nRetrieved Context:\n{context_preview}"}
            ],
            temperature=0.1,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

class SynthesizerAgent:
    def __init__(self, client: AsyncGroq):
        self.client = client

    async def synthesize(self, state: AgentState) -> str:
        context_str = "\n".join(state.retrieved_docs)
        response = await self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are the Lead Synthesizer Agent in an AR-FT swarm architecture. Formulate a definitive, highly technical, and precise final output based strictly on the provided verified context."},
                {"role": "user", "content": f"Original Query: {state.query}\n\nVerified Context:\n{context_str}\n\nCritique Status: {state.critique}"}
            ],
            temperature=0.3,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()

# --- Orchestration Graph Runtime ---

class ARFTMultiAgentSwarm:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.client = AsyncGroq(api_key=api_key)
        self.router = RouterAgent(self.client)
        self.retriever = RetrievalAgent()
        self.critic = CriticAgent(self.client)
        self.synthesizer = SynthesizerAgent(self.client)

    async def run(self, initial_query: str, status_callback) -> str:
        state = AgentState(query=initial_query)
        
        status_callback("⚡ [Router Agent] Optimizing query via Groq LPU (GPT-OSS-120B)...")
        opt_query = await self.router.process(state)
        status_callback(f"⚡ [Router Agent] Optimized Query: `{opt_query}`")
        
        status_callback("⚡ [Retrieval Engine] Fetching context chunks...")
        state.retrieved_docs = await self.retriever.search(opt_query)
        
        status_callback("⚡ [Critic Agent] Evaluating semantic alignment...")
        state.critique = await self.critic.evaluate(state)
        status_callback(f"⚡ [Critic Agent] Status: {state.critique}")
        
        status_callback("⚡ [Synthesizer Agent] Constructing final hardened response...")
        state.final_output = await self.synthesizer.synthesize(state)
        
        return state.final_output

# --- Streamlit UI Layout ---

st.title("QUANTUM-GRAFT // TERMINAL SWARM")
st.markdown("### A.I. Sovereign Adaptive Retrieval-FineTuning Neural Engine")

with st.sidebar:
    st.header("CONFIG // STATUS")
    st.markdown("**Active Engine:** `openai/gpt-oss-120b`")
    st.markdown("**Hardware:** Groq LPU")
    st.markdown("**Topology:** AR-FT Multi-Agent Graph")
    st.markdown("---")
    if os.environ.get("GROQ_API_KEY"):
        st.success("STATUS: API Key Secured via Environment")
    else:
        st.warning("STATUS: GROQ_API_KEY missing in environment variables.")

user_prompt = st.text_area("INITIALIZE SWARM COMMAND:", "Deploy adaptive retrieval fine-tuning for extreme multi-agent synchronization.")

if st.button("EXECUTE SWARM CYCLE"):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("ERROR: GROQ_API_KEY environment variable not detected. Set it in your terminal via `export GROQ_API_KEY='your_key'` before running streamlit.")
    else:
        st.markdown("---")
        status_container = st.empty()
        logs = []

        def update_status(msg):
            logs.append(msg)
            status_container.markdown("\n\n".join([f"> `{log}`" for log in logs]))

        try:
            swarm = ARFTMultiAgentSwarm()
            result = asyncio.run(swarm.run(user_prompt, update_status))
            st.markdown("### EXECUTION RESULT:")
            st.success(result)
        except Exception as e:
            st.error(f"FATAL EXCEPTION: {str(e)}")
