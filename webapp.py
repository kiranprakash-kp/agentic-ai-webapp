
from llama_index.core import (
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent 
from llama_index.core.tools import FunctionTool 
from duckduckgo_search import DDGS
from datetime import datetime
from llama_index.core.prompts import PromptTemplate

import os
import streamlit as st

# Set LLM
llm = None


Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model

# Math functions
def multiply(a:int, b:int) -> int:
    """Multiply two integers and returns the result integer"""
    return a*b

def add(a:int, b:int) -> int:
    """Add two integers and returns the result integer"""
    return a+b

def subtract(a:int, b:int) -> int:
    """Subtract two integers and returns the result integer"""
    return a-b

# Date-Time function
def current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Search function
def search(query:str) -> str: 
    """ 
        Args: 
        query: user prompt 
    return: 
    context (str): search results to the user query 
    """

    req = DDGS() 
    response = req.text(query,max_results=5) 
    context = "" 
    for result in response: 
        context += result['body'] 
    return context

# Convert functions to tools
multiply_tool = FunctionTool.from_defaults(fn=multiply) 
add_tool = FunctionTool.from_defaults(fn=add) 
subtract_tool = FunctionTool.from_defaults(fn=subtract)
time_tool = FunctionTool.from_defaults(fn=current_time)
search_tool = FunctionTool.from_defaults(fn=search)

fntools = [multiply_tool, add_tool,subtract_tool,search_tool,time_tool]

agent = ReActAgent.from_tools(fntools, llm=Settings.llm, 
max_iterations=15,verbose=True)

# Custom prompt template to guide the agent
system_prompt = PromptTemplate(
    "You are an AI assistant with access to tools like search, math operations (add, subtract, multiply), "
    "and current time retrieval. Always use the current time tool for time-related queries and the search tool "
    "for factual queries. Provide clear and complete answers."
)

# Streamlit UI
st.title("AI-Powered Q&A Agent")

st.subheader("Ask factual questions and perform basic math operations like add, subtract, multiply, search, or get the current time!")

user_query = st.text_input("Ask a question:", "")

if st.button("Submit"):
    if user_query:
        
        agent = ReActAgent.from_tools(fntools, llm=Settings.llm, max_iterations=15, verbose=True, system_prompt=system_prompt)
        
        # Spinner while the agent processes
        with st.spinner("Thinking..."):
            answer = agent.chat(user_query)
        
        # Success message and styled output
        st.success("Answer generated successfully!")
        st.markdown(f"### Answer:\n{answer}")
        
        # Optional celebration
        st.balloons()
    else:
        st.warning("Please enter a question.")
































