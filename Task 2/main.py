import asyncio
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import Literal, List, Dict
from collections import deque
from dotenv import load_dotenv

load_dotenv()  

# --- Define state ---
class AgentState(dict):
    user_input: str
    intent: Literal["factual", "creative", "unknown"]
    response: str
    memory: List[Dict[str, str]]


llm = ChatOpenAI(model="gpt-4o-mini")


# --- Intent Detection Node ---
async def detect_intent(state: AgentState) -> AgentState:
    """Classify user input into factual / creative."""
    prompt = f"""
    Classify this user input into either 'factual' or 'creative'.
    Input: "{state['user_input']}"
    """
    result = await llm.ainvoke(prompt)
    text = result.content.lower()
    if "creative" in text:
        state["intent"] = "creative"
    elif "factual" in text:
        state["intent"] = "factual"
    else:
        state["intent"] = "unknown"
    return state


# --- Factual Agent Node ---
async def factual_agent(state: AgentState) -> AgentState:
    memory_str = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in state["memory"]])
    prompt = f"""
    You are a factual assistant. Use the conversation history if needed.

    Conversation history:
    {memory_str}

    Current question: {state['user_input']}
    Answer factually and concisely.
    """
    result = await llm.ainvoke(prompt)
    state["response"] = result.content
    update_memory(state)
    return state


# --- Creative Agent Node ---
async def creative_agent(state: AgentState) -> AgentState:
    memory_str = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in state["memory"]])
    prompt = f"""
    You are a creative assistant. Use the conversation history if needed.

    Conversation history:
    {memory_str}

    Current prompt: {state['user_input']}
    Generate a creative, imaginative response.
    """
    result = await llm.ainvoke(prompt)
    state["response"] = result.content
    update_memory(state)
    return state


# --- Update Memory Helper ---
def update_memory(state: AgentState, max_memory: int = 3):
    """Keep last few interactions (2–3 turns)."""
    memory = deque(state["memory"], maxlen=max_memory)
    memory.append({"user": state["user_input"], "assistant": state["response"]})
    state["memory"] = list(memory)
    


# --- Router function ---
def route_by_intent(state: AgentState) -> str:
    if state["intent"] == "factual":
        return "factual_agent"
    elif state["intent"] == "creative":
        return "creative_agent"
    else:
        return END


# --- Build Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("detect_intent", detect_intent)
workflow.add_node("factual_agent", factual_agent)
workflow.add_node("creative_agent", creative_agent)

workflow.set_entry_point("detect_intent")
workflow.add_conditional_edges("detect_intent", route_by_intent, {
    "factual_agent": "factual_agent",
    "creative_agent": "creative_agent",
})

workflow.add_edge("factual_agent", END)
workflow.add_edge("creative_agent", END)

app = workflow.compile()

# print(app.get_graph().draw_ascii())

# --- Example Run with Async ---
async def main():
    state = {"user_input": "", "memory": []}

    while True:
         print("Type 'exit' or 'quit' to stop.")
         user_input = input("👤: ")
         if user_input.lower() in ["exit", "quit"]:
             break
         
         state["user_input"] = user_input
         result = await app.ainvoke(state)
         print("🤖:", result["response"], '\n')
         state["memory"] = result["memory"]



if __name__ == "__main__":
    asyncio.run(main())
