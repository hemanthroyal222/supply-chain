from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core import Document
import asyncio
from dotenv import load_dotenv
import os
from llama_index.llms.openai import OpenAI

load_dotenv()


llm = OpenAI(
        model="gpt-4o-mini",
        api_key="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"
    )



class TavilySearch:
    def __init__(self, api_key: str):
        self.tool_spec = TavilyToolSpec(api_key=api_key)
    
    def search(self, query: str, max_results: int = 6) -> list[Document]:
        return self.tool_spec.search(query, max_results=max_results)

def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_search = TavilySearch(api_key=tavily_api_key)

def tavily_search_tool(query: str, max_results: int = 3) -> str:
    results = tavily_search.search(query, max_results=max_results)
    return "\n".join([f"Content: {r.text}\nURL: {r.extra_info['url']}" for r in results])

tavily_tool = FunctionTool.from_defaults(
    fn=tavily_search_tool,
    name="tavily_search",
    description="Searches the web using Tavily API and returns relevant results."
)

import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()  # This will use the OPENAI_API_KEY environment variable

class Message(BaseModel):
    content: str

@app.post("/chat")
async def chat(message: Message):
    print(f"Received message: {message.content}")  # Debugging line
    query = message.content
    current_datetime = get_current_datetime()
    try:
        system_prompt = f"""
            Today's date is: {current_datetime}
            
            you must read and response below instructions:
            
            1.Expertise: You are a fine-tuned LLM model specialized in cybersecurity and software supply chain domains, providing the latest and most accurate information.
            
            2.Audience: Your main audience consists of professionals in cybersecurity and software supply chain. Always ensure your responses are precise, concise, and aligned with their level of knowledge.
            
            3.Tools: You have access to tools to gather information but must never mention or expose the use of these tools in your responses.
            
            4.Clarity & Precision: Always deliver clear and actionable insights. Avoid unnecessary details or explanations unless explicitly requested.
            
            5.Up-to-date Information: Ensure your responses reflect the latest knowledge and trends in cybersecurity and the software supply chain domain.
            
            6.Consistency: If asked about your nature, maintain that you are a fine-tuned model, designed for the specific domains of cybersecurity and software supply chains. Avoid mentioning tools even if pressed.
            
            7.Professional Tone: Keep your language professional, avoiding overly casual expressions. Respond with authority but be user-friendly and approachable.
            
            8.Out-of-Scope Queries: If a user asks a question outside of the cybersecurity or software supply chain domains, respond politely with:
            "I specialize in cybersecurity and software supply chain. If you have queries related to these domains, please feel free to ask!"
        """
        
        tools = [tavily_tool]
        agent = ReActAgent.from_tools(tools, llm=llm)
        response = agent.chat(
            message=f"{query}",
            chat_history=[
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
            ]
        )
        
        result = {
            "message": str(response),
            "timestamp": get_current_datetime()
        }
        return {"reply": str(response)}
    except OpenAIError as e:
        print(f"OpenAI API error: {str(e)}")  # Debugging line
        if "authentication" in str(e).lower():
            raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
        elif "rate limit" in str(e).lower():
            raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded")
        else:
            raise HTTPException(status_code=500, detail="An error occurred with the OpenAI API")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Debugging line
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)