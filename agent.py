
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
import os


llm = None


Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model

def multiply(a:int, b:int) -> int:
    """Multiply two integers and returns the result integer"""
    return a*b

def add(a:int, b:int) -> int:
    """Add two integers and returns the result integer"""
    return a+b

def subtract(a:int, b:int) -> int:
    """Subtract two integers and returns the result integer"""
    return a-b

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

multiply_tool = FunctionTool.from_defaults(fn=multiply) 
add_tool = FunctionTool.from_defaults(fn=add) 
subtract_tool = FunctionTool.from_defaults(fn=subtract) 
search_tool = FunctionTool.from_defaults(fn=search)

fntools = [multiply_tool, add_tool,subtract_tool,search_tool]

agent = ReActAgent.from_tools(fntools, llm=Settings.llm, 
max_iterations=15,verbose=True)

response=agent.chat("Where is Mount Everest located, what is its height in meters, and what is that height multiplied by 2?") 
print(response)






























