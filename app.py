import asyncio
import os
from typing import List
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# --- Schemas & State ---
class AgentState(BaseModel):
    query: str
    retrieved_docs: List[str] = Field(default_factory=list)
    critique: str = ""
    final_output: str = ""
    iteration: int = 0

# --- Live OpenAI gpt-oss-120b Powered Agents ---

class RouterAgent:
    """Agent Alpha: Analyzes query and optimizes search parameters using OpenAI gpt-oss-120b."""
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def process(self, state: AgentState) -> str:
        print(f"[Router Agent] Optimizing query via gpt-oss-120b: '{state.query}'")
        response = await self.client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a specialized query optimization agent for high-precision hybrid retrieval systems. Output ONLY the optimized search query string without any conversational filler."},
                {"role": "user", "content": f"Optimize this query for vector search: {state.query}"}
            ],
            temperature=0.1,
            max_tokens=60
        )
        optimized = response.choices[0].message.content.strip()
        print(f"[Router Agent] Optimized Query: {optimized}")
        return optimized

class RetrievalAgent:
    """Performs Hybrid BM25 + Vector Search (Production Mock / Hook point)."""
    async def search(self, optimized_query: str) -> List[str]:
        print(f"[Retrieval Engine] Fetching context for: {optimized_query}")
        await asyncio.sleep(0.1) # Simulate high-speed vector DB fetch (e.g., Qdrant)
        return [
            "Context Chunk 1: AR-FT combines adaptive retrieval thresholds with domain-specific QLoRA weights to mitigate hallucination drift.",
            "Context Chunk 2: Multi-agent coordination graphs utilize state-machine loops to validate semantic alignment before final output synthesis."
        ]

class CriticAgent:
    """Agent Beta: Evaluates retrieved context and intermediate alignment using gpt-oss-120b."""
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def evaluate(self, state: AgentState) -> str:
        print("[Critic Agent] Evaluating retrieved chunks via gpt-oss-120b...")
        context_preview = "\n".join(state.retrieved_docs)
        response = await self.client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a strict validation critic agent. Evaluate if the retrieved context adequately answers the initial user query. Respond with 'Approved: [Reason]' or 'Rejected: [Reason]'."},
                {"role": "user", "content": f"Query: {state.query}\n\nRetrieved Context:\n{context_preview}"}
            ],
            temperature=0.1,
            max_tokens=100
        )
        critique_result = response.choices[0].message.content.strip()
        print(f"[Critic Agent] Result: {critique_result}")
        return critique_result

class SynthesizerAgent:
    """Agent Gamma: Generates final hardened response via gpt-oss-120b."""
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def synthesize(self, state: AgentState) -> str:
        print("[Synthesizer Agent] Constructing final response via gpt-oss-120b...")
        context_str = "\n".join(state.retrieved_docs)
        response = await self.client.chat.completions.create(
            model="gpt-oss-120b",
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
        # Initialize Async OpenAI Client (picks up OPENAI_API_KEY env variable)
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.router = RouterAgent(self.client)
        self.retriever = RetrievalAgent()
        self.critic = CriticAgent(self.client)
        self.synthesizer = SynthesizerAgent(self.client)

    async def run(self, initial_query: str) -> str:
        state = AgentState(query=initial_query)
        
        # Step 1: Route & Optimize Query via gpt-oss-120b
        opt_query = await self.router.process(state)
        
        # Step 2: Retrieve Context Chunks
        state.retrieved_docs = await self.retriever.search(opt_query)
        
        # Step 3: Critique & Validate Retrieval via gpt-oss-120b
        state.critique = await self.critic.evaluate(state)
        
        # Step 4: Synthesize Final Output via gpt-oss-120b
        state.final_output = await self.synthesizer.synthesize(state)
        
        return state.final_output

# --- Execution Entrypoint ---
if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Please set your OPENAI_API_KEY environment variable before running.")
        
    swarm = ARFTMultiAgentSwarm()
    user_prompt = "Deploy adaptive retrieval fine-tuning for extreme multi-agent synchronization."
    
    result = asyncio.run(swarm.run(user_prompt))
    print("\n" + "="*40 + "\n" + result + "\n" + "="*40)
