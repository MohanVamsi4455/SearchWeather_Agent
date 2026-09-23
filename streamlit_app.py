import os

import certifi
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# Same "ReAct" template as the "hwchase17/react" LangChain hub prompt, inlined
# so deployment doesn't depend on LangSmith's public-prompt-pull safety gate
# (langsmith now requires dangerously_pull_public_prompt=True for hub pulls).
REACT_PROMPT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

# Load GROQ_API_KEY, TAVILY_API_KEY, WEATHER_STACK_API_KEY, etc. from .env
load_dotenv()
# Fixes SSL certificate verification errors on some Windows/corporate setups
os.environ["SSL_CERT_FILE"] = certifi.where()

st.set_page_config(page_title="ReAct Agent", page_icon="🤖")
st.title("🤖 ReAct Agent (Groq + Tavily)")


# The docstring is not just documentation here — LangChain reads it as the
# tool's description, which is what the LLM sees when deciding whether to call it.
@tool
def get_weather(city: str) -> dict:
    """Get the current weather information for a given city.

    Use this tool when the user asks about current weather,
    temperature, humidity, or weather conditions in a city.
    """
    url = (
        "https://api.weatherstack.com/current"
        f"?access_key={os.getenv('WEATHER_STACK_API_KEY')}&query={city}"
    )
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Weatherstack returns HTTP 200 even on failure; errors show up in the body
    if "error" in data:
        raise Exception(data["error"]["info"])

    return data


# @st.cache_resource keeps the LLM, tools, and agent alive across reruns —
# without it, Streamlit would rebuild everything on every chat message.
@st.cache_resource
def build_agent_executor():
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    search_tool = TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)
    tools = [search_tool, get_weather]

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="qwen/qwen3.8-27b",
        temperature=0.1,
        # Groq's on-demand tier caps this model at 1000 output tokens/minute;
        # without a cap, requests can be estimated above that and get rate-limited.
        max_tokens=512,
    )

    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

    # create_react_agent needs a plain chat model here, not one wrapped with
    # with_structured_output(), since the ReAct loop passes a `stop` sequence
    # into each LLM call to end generation before a fake Observation is produced.
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor = build_agent_executor()

# Chat history persists per browser session (reset on page refresh)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw past turns on every rerun, since Streamlit reruns the whole script each time
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask me anything (e.g. weather, search queries)...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Runs the full ReAct loop: LLM decides which tool to call (if any),
            # tool result is fed back in, repeated until a Final Answer is reached.
            response = agent_executor.invoke({"input": user_input})
            answer = response["output"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
