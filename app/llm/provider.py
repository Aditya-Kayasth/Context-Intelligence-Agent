"""
LLM provider using LangChain.
Supports Groq, Ollama, OpenAI, Anthropic, Grok, and MCP options.
Integrated with LangSmith and LangGraph capabilities.
"""
from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from app.config import settings

logger = logging.getLogger(__name__)

class BaseLLMProvider:
    def __init__(self, llm):
        self.llm = llm

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        # Adding support for json format if needed could be passed to llm
        try:
            resp = await self.llm.ainvoke(messages)
            return resp.content
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            raise

def get_llm_provider() -> BaseLLMProvider:
    explicit = settings.llm_provider.lower()
    
    if explicit == "mcp":
        logger.info("LLM provider: MCP (Model Context Protocol)")
        llm = ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key)
        return BaseLLMProvider(llm)

    if explicit == "grok":
        logger.info("LLM provider: Grok")
        llm = ChatOpenAI(
            api_key=settings.xai_api_key, 
            base_url="https://api.x.ai/v1", 
            model="grok-2-latest", 
            temperature=0.0
        )
        return BaseLLMProvider(llm)
        
    if explicit == "anthropic":
        logger.info("LLM provider: Anthropic (explicit)")
        llm = ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model_name=settings.claude_model,
            temperature=0.0
        )
        return BaseLLMProvider(llm)
        
    if explicit == "openai":
        logger.info("LLM provider: OpenAI (explicit)")
        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model_name=settings.openai_model,
            temperature=0.0
        )
        return BaseLLMProvider(llm)

    if explicit == "ollama":
        logger.info("LLM provider: Ollama (explicit)")
        llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model, 
            temperature=0.0
        )
        return BaseLLMProvider(llm)

    # Auto fallback
    if getattr(settings, "groq_api_key", None) and explicit != "ollama":
        logger.info("LLM provider: Groq (auto-selected)")
        llm = ChatGroq(
            api_key=settings.groq_api_key, 
            model_name=settings.groq_model, 
            temperature=0.0
        )
        return BaseLLMProvider(llm)
        
    logger.info("LLM provider: Ollama (default)")
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model, 
        temperature=0.0
    )
    return BaseLLMProvider(llm)
