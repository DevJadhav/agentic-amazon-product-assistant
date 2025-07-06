"""
LLM Manager for the AI-Powered Amazon Product Assistant.
Handles LLM provider initialization, configuration, and execution.
"""

import time
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from functools import lru_cache

import streamlit as st
from langsmith import traceable
from openai import OpenAI
from groq import Groq
from google import genai
import ollama

from core.config import config


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def create_client(self) -> Any:
        """Create and return the LLM client."""
        pass
    
    @abstractmethod
    def generate_response(self, client: Any, messages: List[Dict[str, str]], 
                        **kwargs) -> str:
        """Generate response from the LLM."""
        pass
    
    @property
    def supports_top_k(self) -> bool:
        """Check if provider supports top_k parameter."""
        return False


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self):
        super().__init__("OpenAI")
    
    def create_client(self) -> OpenAI:
        """Create OpenAI client."""
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured")
        return OpenAI(api_key=config.OPENAI_API_KEY.get_secret_value())
    
    def generate_response(self, client: OpenAI, messages: List[Dict[str, str]], 
                        model: str, temperature: float, max_tokens: int, 
                        top_p: float, **kwargs) -> str:
        """Generate response using OpenAI."""
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return completion.choices[0].message.content


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider."""
    
    def __init__(self):
        super().__init__("Groq")
    
    def create_client(self) -> Groq:
        """Create Groq client."""
        if not config.GROQ_API_KEY:
            raise ValueError("Groq API key not configured")
        return Groq(api_key=config.GROQ_API_KEY.get_secret_value())
    
    def generate_response(self, client: Groq, messages: List[Dict[str, str]], 
                        model: str, temperature: float, max_tokens: int, 
                        top_p: float, **kwargs) -> str:
        """Generate response using Groq."""
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return completion.choices[0].message.content


class GoogleProvider(BaseLLMProvider):
    """Google GenAI LLM provider."""
    
    def __init__(self):
        super().__init__("Google")
    
    @property
    def supports_top_k(self) -> bool:
        """Google supports top_k parameter."""
        return True
    
    def create_client(self) -> genai.Client:
        """Create Google GenAI client."""
        if not config.GOOGLE_API_KEY:
            raise ValueError("Google API key not configured")
        return genai.Client(api_key=config.GOOGLE_API_KEY.get_secret_value())
    
    def generate_response(self, client: genai.Client, messages: List[Dict[str, str]], 
                        model: str, temperature: float, max_tokens: int, 
                        top_p: float, top_k: int = 40, **kwargs) -> str:
        """Generate response using Google GenAI."""
        # Convert messages to Google format
        google_messages = []
        for message in messages:
            if message.get("content") and message["content"].strip():
                google_role = "user" if message["role"] == "user" else "model"
                google_messages.append({
                    "role": google_role,
                    "parts": [{"text": message["content"]}]
                })
        
        response = client.models.generate_content(
            model=model,
            contents=google_messages,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k
            }
        )
        return response.text


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local models."""
    
    def __init__(self):
        super().__init__("Ollama")
    
    def create_client(self) -> ollama.Client:
        """Create Ollama client."""
        return ollama.Client(host=config.OLLAMA_BASE_URL)
    
    def generate_response(self, client: ollama.Client, messages: List[Dict[str, str]], 
                        model: str, temperature: float, max_tokens: int, 
                        **kwargs) -> str:
        """Generate response using Ollama."""
        response = client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        return response["message"]["content"]


class LLMManager:
    """Manages LLM providers and handles response generation."""
    
    # Provider registry
    PROVIDERS = {
        "OpenAI": OpenAIProvider,
        "Groq": GroqProvider,
        "Google": GoogleProvider,
        "Ollama": OllamaProvider
    }
    
    # Model configurations
    MODEL_CONFIGS = {
        "OpenAI": {
            "default": "gpt-4o-mini",
            "models": ["gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        },
        "Groq": {
            "default": "llama3-70b-8192",
            "models": ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
        },
        "Google": {
            "default": "gemini-1.5-flash",
            "models": ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro", "gemini-1.0-pro"]
        },
        "Ollama": {
            "default": "llama3.2",
            "models": ["llama3.2", "llama3.2:1b", "gemma2:2b", "qwen2.5:3b", "phi3.5", "mistral"]
        }
    }
    
    def __init__(self):
        """Initialize LLM Manager."""
        self._provider_cache = {}
        self._client_cache = {}
    
    @lru_cache(maxsize=4)
    def get_provider(self, provider_name: str) -> BaseLLMProvider:
        """Get LLM provider instance (cached)."""
        if provider_name not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider_class = self.PROVIDERS[provider_name]
        return provider_class()
    
    def get_client(self, provider_name: str) -> Any:
        """Get or create LLM client for provider."""
        if provider_name not in self._client_cache:
            provider = self.get_provider(provider_name)
            self._client_cache[provider_name] = provider.create_client()
        return self._client_cache[provider_name]
    
    def get_available_models(self, provider_name: str) -> List[str]:
        """Get available models for a provider."""
        return self.MODEL_CONFIGS.get(provider_name, {}).get("models", [])
    
    def get_default_model(self, provider_name: str) -> str:
        """Get default model for a provider."""
        return self.MODEL_CONFIGS.get(provider_name, {}).get("default", "")
    
    @traceable
    def generate_response(self, provider_name: str, messages: List[Dict[str, str]], 
                        model: str, temperature: float = 0.7, 
                        max_tokens: int = 500, top_p: float = 1.0, 
                        top_k: int = 40) -> Dict[str, Any]:
        """Generate response from LLM with comprehensive tracking."""
        start_time = time.time()
        
        # Prepare metadata
        metadata = {
            "provider": provider_name,
            "model": model,
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k if provider_name == "Google" else None
            },
            "message_count": len(messages),
            "total_chars": sum(len(msg.get("content", "")) for msg in messages)
        }
        
        try:
            # Get provider and client
            provider = self.get_provider(provider_name)
            client = self.get_client(provider_name)
            
            # Generate response
            kwargs = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
            
            if provider.supports_top_k:
                kwargs["top_k"] = top_k
            
            response = provider.generate_response(client, messages, **kwargs)
            
            response_time = time.time() - start_time
            
            return {
                "status": "success",
                "response": response,
                "response_length": len(response),
                "response_time_ms": round(response_time * 1000, 2),
                "metadata": metadata
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "response_time_ms": round(response_time * 1000, 2),
                "metadata": metadata
            }
    
    def update_provider_stats(self, provider_name: str, model: str, 
                            response_time_ms: float, use_rag: bool) -> None:
        """Update provider performance statistics."""
        if 'provider_model_stats' not in st.session_state:
            st.session_state.provider_model_stats = {}
        
        key = f"{provider_name}::{model}"
        
        if key not in st.session_state.provider_model_stats:
            st.session_state.provider_model_stats[key] = {
                "provider": provider_name,
                "model": model,
                "total_queries": 0,
                "total_time_ms": 0,
                "total_rag_time_ms": 0,
                "total_llm_time_ms": 0,
                "rag_queries": 0,
                "non_rag_queries": 0,
                "min_llm_time_ms": float('inf'),
                "max_llm_time_ms": 0,
                "recent_performances": []
            }
        
        stats = st.session_state.provider_model_stats[key]
        
        # Update statistics
        stats["total_queries"] += 1
        stats["total_time_ms"] += response_time_ms
        stats["total_llm_time_ms"] += response_time_ms
        
        if use_rag:
            stats["rag_queries"] += 1
        else:
            stats["non_rag_queries"] += 1
        
        stats["min_llm_time_ms"] = min(stats["min_llm_time_ms"], response_time_ms)
        stats["max_llm_time_ms"] = max(stats["max_llm_time_ms"], response_time_ms)
        
        # Track recent performances
        stats["recent_performances"].append({
            "timestamp": time.time(),
            "llm_time_ms": response_time_ms,
            "use_rag": use_rag
        })
        
        # Keep only last 50 performances
        if len(stats["recent_performances"]) > 50:
            stats["recent_performances"] = stats["recent_performances"][-50:]


# Global LLM manager instance
llm_manager = LLMManager() 