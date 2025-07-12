"""
Enhanced LLM Manager with Multi-Provider Support
Comprehensive error handling, rate limiting, fallback mechanisms, and performance optimization.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import hashlib
import os
from abc import ABC, abstractmethod

# Third-party imports
from openai import OpenAI, AsyncOpenAI
from groq import Groq
from google import genai
import ollama
from langsmith import traceable
import backoff

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GROQ = "groq"
    GOOGLE = "google"
    OLLAMA = "ollama"

class ResponseStatus(Enum):
    """Response status indicators."""
    SUCCESS = "success"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    QUOTA_EXCEEDED = "quota_exceeded"

@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider."""
    provider: LLMProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: List[str] = field(default_factory=list)
    default_model: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30
    rate_limit_per_minute: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority

@dataclass
class RequestMetrics:
    """Metrics for tracking request performance."""
    provider: LLMProvider
    model: str
    timestamp: datetime
    response_time: float
    token_count: int
    status: ResponseStatus
    error_message: Optional[str] = None

@dataclass
class RateLimitInfo:
    """Rate limiting information."""
    requests_per_minute: int
    requests_made: deque = field(default_factory=deque)
    last_reset: datetime = field(default_factory=datetime.now)

class LLMProviderInterface(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              model: str, 
                              **kwargs) -> Dict[str, Any]:
        """Generate response from the provider."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        pass

class OpenAIProvider(LLMProviderInterface):
    """OpenAI provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self.async_client = AsyncOpenAI(api_key=config.api_key)
    
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              model: str, 
                              **kwargs) -> Dict[str, Any]:
        """Generate response using OpenAI API."""
        try:
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                timeout=self.config.timeout
            )
            
            return {
                'content': response.choices[0].message.content,
                'model': response.model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'finish_reason': response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available OpenAI models."""
        return ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k']
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text.split()) // 0.75  # Rough estimation

class GroqProvider(LLMProviderInterface):
    """Groq provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = Groq(api_key=config.api_key)
    
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              model: str, 
                              **kwargs) -> Dict[str, Any]:
        """Generate response using Groq API."""
        try:
            # Groq doesn't have async client, so we'll run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                    temperature=kwargs.get('temperature', self.config.temperature)
                )
            )
            
            return {
                'content': response.choices[0].message.content,
                'model': response.model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'finish_reason': response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available Groq models."""
        return ['mixtral-8x7b-32768', 'llama2-70b-4096', 'gemma-7b-it']
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text.split()) // 0.75

class GoogleProvider(LLMProviderInterface):
    """Google Gemini provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = genai.Client(api_key=config.api_key)
    
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              model: str, 
                              **kwargs) -> Dict[str, Any]:
        """Generate response using Google Gemini API."""
        try:
            # Convert messages to Gemini format
            prompt = self._convert_messages_to_prompt(messages)
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    generation_config={
                        'max_output_tokens': kwargs.get('max_tokens', self.config.max_tokens),
                        'temperature': kwargs.get('temperature', self.config.temperature)
                    }
                )
            )
            
            return {
                'content': response.text,
                'model': model,
                'usage': {
                    'prompt_tokens': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    'completion_tokens': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                    'total_tokens': response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                },
                'finish_reason': 'stop'
            }
            
        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to single prompt for Gemini."""
        prompt_parts = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if role == 'system':
                prompt_parts.append(f"Instructions: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def get_available_models(self) -> List[str]:
        """Get available Google models."""
        return ['gemini-pro', 'gemini-pro-vision']
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text.split()) // 0.75

class OllamaProvider(LLMProviderInterface):
    """Ollama provider implementation."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = ollama.Client(host=config.base_url or 'http://localhost:11434')
    
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              model: str, 
                              **kwargs) -> Dict[str, Any]:
        """Generate response using Ollama API."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat(
                    model=model,
                    messages=messages,
                    options={
                        'temperature': kwargs.get('temperature', self.config.temperature),
                        'num_predict': kwargs.get('max_tokens', self.config.max_tokens)
                    }
                )
            )
            
            return {
                'content': response['message']['content'],
                'model': model,
                'usage': {
                    'prompt_tokens': response.get('prompt_eval_count', 0),
                    'completion_tokens': response.get('eval_count', 0),
                    'total_tokens': response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
                },
                'finish_reason': 'stop'
            }
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available Ollama models."""
        try:
            models = self.client.list()
            return [model['name'] for model in models['models']]
        except:
            return ['llama2', 'mistral', 'codellama']
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text.split()) // 0.75

class EnhancedLLMManager:
    """Enhanced LLM manager with multi-provider support, error handling, and optimization."""
    
    def __init__(self, provider_configs: List[ProviderConfig]):
        """Initialize the enhanced LLM manager."""
        self.provider_configs = {config.provider: config for config in provider_configs}
        self.providers: Dict[LLMProvider, LLMProviderInterface] = {}
        self.rate_limits: Dict[LLMProvider, RateLimitInfo] = {}
        self.metrics: List[RequestMetrics] = []
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.fallback_chain: List[LLMProvider] = []
        
        # Performance tracking
        self.performance_stats = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'avg_response_time': 0.0,
            'error_rate': 0.0
        })
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize providers
        self._initialize_providers()
        self._setup_fallback_chain()
        
        logger.info(f"Enhanced LLM Manager initialized with {len(self.providers)} providers")
    
    def _initialize_providers(self):
        """Initialize all enabled providers."""
        provider_classes = {
            LLMProvider.OPENAI: OpenAIProvider,
            LLMProvider.GROQ: GroqProvider,
            LLMProvider.GOOGLE: GoogleProvider,
            LLMProvider.OLLAMA: OllamaProvider
        }
        
        for provider_type, config in self.provider_configs.items():
            if not config.enabled:
                continue
                
            try:
                provider_class = provider_classes.get(provider_type)
                if provider_class:
                    self.providers[provider_type] = provider_class(config)
                    self.rate_limits[provider_type] = RateLimitInfo(
                        requests_per_minute=config.rate_limit_per_minute
                    )
                    logger.info(f"Initialized {provider_type.value} provider")
                
            except Exception as e:
                logger.error(f"Failed to initialize {provider_type.value}: {e}")
    
    def _setup_fallback_chain(self):
        """Setup fallback chain based on provider priorities."""
        enabled_providers = [
            (provider, config) for provider, config in self.provider_configs.items() 
            if config.enabled and provider in self.providers
        ]
        
        # Sort by priority (lower number = higher priority)
        enabled_providers.sort(key=lambda x: x[1].priority)
        
        self.fallback_chain = [provider for provider, _ in enabled_providers]
        logger.info(f"Fallback chain: {[p.value for p in self.fallback_chain]}")
    
    def _check_rate_limit(self, provider: LLMProvider) -> bool:
        """Check if provider is within rate limits."""
        with self.lock:
            rate_info = self.rate_limits.get(provider)
            if not rate_info:
                return True
            
            now = datetime.now()
            
            # Remove requests older than 1 minute
            while rate_info.requests_made and rate_info.requests_made[0] < now - timedelta(minutes=1):
                rate_info.requests_made.popleft()
            
            # Check if under limit
            return len(rate_info.requests_made) < rate_info.requests_per_minute
    
    def _record_request(self, provider: LLMProvider):
        """Record a request for rate limiting."""
        with self.lock:
            rate_info = self.rate_limits.get(provider)
            if rate_info:
                rate_info.requests_made.append(datetime.now())
    
    def _get_cache_key(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        """Generate cache key for request."""
        cache_data = {
            'messages': messages,
            'model': model,
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1000)
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if response is cached and valid."""
        with self.lock:
            if cache_key in self.response_cache:
                cached_response, timestamp = self.response_cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                    return cached_response
                else:
                    del self.response_cache[cache_key]
        
        return None
    
    def _update_cache(self, cache_key: str, response: Dict[str, Any]):
        """Update cache with new response."""
        with self.lock:
            # Limit cache size
            if len(self.response_cache) > 1000:
                oldest_key = min(self.response_cache.keys(), 
                               key=lambda k: self.response_cache[k][1])
                del self.response_cache[oldest_key]
            
            self.response_cache[cache_key] = (response, datetime.now())
    
    def _record_metrics(self, 
                       provider: LLMProvider, 
                       model: str, 
                       response_time: float, 
                       token_count: int, 
                       status: ResponseStatus, 
                       error_message: Optional[str] = None):
        """Record request metrics."""
        metrics = RequestMetrics(
            provider=provider,
            model=model,
            timestamp=datetime.now(),
            response_time=response_time,
            token_count=token_count,
            status=status,
            error_message=error_message
        )
        
        self.metrics.append(metrics)
        
        # Update performance stats
        stats = self.performance_stats[provider]
        stats['total_requests'] += 1
        
        if status == ResponseStatus.SUCCESS:
            stats['successful_requests'] += 1
            
            # Update average response time
            old_avg = stats['avg_response_time']
            total_successful = stats['successful_requests']
            stats['avg_response_time'] = (old_avg * (total_successful - 1) + response_time) / total_successful
        
        # Update error rate
        stats['error_rate'] = 1.0 - (stats['successful_requests'] / stats['total_requests'])
        
        # Keep only recent metrics (last 1000)
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=60
    )
    async def _make_request_with_retry(self, 
                                     provider: LLMProvider, 
                                     messages: List[Dict[str, str]], 
                                     model: str, 
                                     **kwargs) -> Dict[str, Any]:
        """Make request with exponential backoff retry."""
        provider_instance = self.providers[provider]
        return await provider_instance.generate_response(messages, model, **kwargs)
    
    @traceable
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              preferred_provider: Optional[LLMProvider] = None,
                              model: Optional[str] = None,
                              enable_cache: bool = True,
                              enable_fallback: bool = True,
                              **kwargs) -> Dict[str, Any]:
        """Generate response with multi-provider support and fallback."""
        
        # Determine provider order
        if preferred_provider and preferred_provider in self.fallback_chain:
            provider_order = [preferred_provider] + [p for p in self.fallback_chain if p != preferred_provider]
        else:
            provider_order = self.fallback_chain.copy()
        
        # Check cache first
        if enable_cache:
            cache_key = self._get_cache_key(messages, model or 'default', **kwargs)
            cached_response = self._check_cache(cache_key)
            if cached_response:
                logger.debug("Cache hit for request")
                return {**cached_response, 'from_cache': True}
        
        last_error = None
        
        for provider in provider_order:
            if provider not in self.providers:
                continue
            
            # Check rate limits
            if not self._check_rate_limit(provider):
                logger.warning(f"Rate limit exceeded for {provider.value}")
                continue
            
            try:
                # Get provider config and model
                config = self.provider_configs[provider]
                used_model = model or config.default_model or config.models[0] if config.models else 'default'
                
                # Record request for rate limiting
                self._record_request(provider)
                
                # Make request
                start_time = time.time()
                response = await self._make_request_with_retry(
                    provider, messages, used_model, **kwargs
                )
                response_time = time.time() - start_time
                
                # Record successful metrics
                token_count = response.get('usage', {}).get('total_tokens', 0)
                self._record_metrics(
                    provider, used_model, response_time, token_count, ResponseStatus.SUCCESS
                )
                
                # Enhance response with metadata
                enhanced_response = {
                    **response,
                    'provider': provider.value,
                    'response_time': response_time,
                    'from_cache': False,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Update cache
                if enable_cache:
                    self._update_cache(cache_key, enhanced_response)
                
                logger.info(f"Successful response from {provider.value} in {response_time:.2f}s")
                return enhanced_response
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Determine error type
                if 'rate limit' in error_msg.lower():
                    status = ResponseStatus.RATE_LIMITED
                elif 'quota' in error_msg.lower():
                    status = ResponseStatus.QUOTA_EXCEEDED
                elif 'timeout' in error_msg.lower():
                    status = ResponseStatus.TIMEOUT
                else:
                    status = ResponseStatus.ERROR
                
                # Record failed metrics
                self._record_metrics(
                    provider, used_model, 0.0, 0, status, error_msg
                )
                
                logger.error(f"Request failed for {provider.value}: {e}")
                
                if not enable_fallback:
                    break
        
        # All providers failed
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get comprehensive provider statistics."""
        stats = {}
        
        for provider in self.providers:
            provider_metrics = [m for m in self.metrics if m.provider == provider]
            
            if provider_metrics:
                response_times = [m.response_time for m in provider_metrics if m.status == ResponseStatus.SUCCESS]
                error_count = len([m for m in provider_metrics if m.status != ResponseStatus.SUCCESS])
                
                stats[provider.value] = {
                    'total_requests': len(provider_metrics),
                    'successful_requests': len(response_times),
                    'error_count': error_count,
                    'error_rate': error_count / len(provider_metrics) if provider_metrics else 0,
                    'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                    'min_response_time': min(response_times) if response_times else 0,
                    'max_response_time': max(response_times) if response_times else 0,
                    'current_rate_limit_usage': len(self.rate_limits[provider].requests_made) if provider in self.rate_limits else 0,
                    'rate_limit_per_minute': self.provider_configs[provider].rate_limit_per_minute,
                    'enabled': self.provider_configs[provider].enabled,
                    'priority': self.provider_configs[provider].priority
                }
            else:
                stats[provider.value] = {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'error_count': 0,
                    'error_rate': 0,
                    'avg_response_time': 0,
                    'enabled': self.provider_configs[provider].enabled,
                    'priority': self.provider_configs[provider].priority
                }
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the LLM manager."""
        total_providers = len(self.providers)
        healthy_providers = 0
        
        for provider in self.providers:
            provider_stats = self.performance_stats[provider]
            
            # Consider provider healthy if error rate < 50% and it has processed requests
            if (provider_stats['total_requests'] == 0 or 
                provider_stats['error_rate'] < 0.5):
                healthy_providers += 1
        
        health_percentage = (healthy_providers / total_providers) * 100 if total_providers > 0 else 0
        
        return {
            'status': 'healthy' if health_percentage >= 50 else 'degraded' if health_percentage > 0 else 'unhealthy',
            'healthy_providers': healthy_providers,
            'total_providers': total_providers,
            'health_percentage': health_percentage,
            'fallback_chain': [p.value for p in self.fallback_chain],
            'cache_size': len(self.response_cache),
            'total_requests': sum(stats['total_requests'] for stats in self.performance_stats.values())
        }
    
    def clear_cache(self):
        """Clear response cache."""
        with self.lock:
            self.response_cache.clear()
            logger.info("Response cache cleared")
    
    def reset_metrics(self):
        """Reset all metrics and performance stats."""
        with self.lock:
            self.metrics.clear()
            self.performance_stats.clear()
            logger.info("Metrics reset")
    
    def update_provider_config(self, provider: LLMProvider, **kwargs):
        """Update provider configuration."""
        if provider in self.provider_configs:
            config = self.provider_configs[provider]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"Updated {provider.value} configuration: {kwargs}")
    
    def get_available_models(self, provider: LLMProvider) -> List[str]:
        """Get available models for a provider."""
        if provider in self.providers:
            return self.providers[provider].get_available_models()
        return []
    
    def estimate_cost(self, 
                     provider: LLMProvider, 
                     model: str, 
                     prompt_tokens: int, 
                     completion_tokens: int) -> float:
        """Estimate cost for a request (placeholder - implement with actual pricing)."""
        # This would be implemented with actual pricing data
        cost_per_1k_tokens = {
            LLMProvider.OPENAI: {'gpt-4': 0.03, 'gpt-3.5-turbo': 0.002},
            LLMProvider.GROQ: {'mixtral-8x7b-32768': 0.0007},
            LLMProvider.GOOGLE: {'gemini-pro': 0.001},
            LLMProvider.OLLAMA: {}  # Usually free/self-hosted
        }
        
        provider_costs = cost_per_1k_tokens.get(provider, {})
        cost_per_token = provider_costs.get(model, 0.001) / 1000
        
        return (prompt_tokens + completion_tokens) * cost_per_token


def create_enhanced_llm_manager() -> EnhancedLLMManager:
    """Create and configure the enhanced LLM manager."""
    
    # Load configurations from environment or config files
    provider_configs = [
        ProviderConfig(
            provider=LLMProvider.OPENAI,
            api_key=os.getenv('OPENAI_API_KEY'),
            models=['gpt-4', 'gpt-3.5-turbo'],
            default_model='gpt-4',
            priority=1,
            rate_limit_per_minute=60,
            enabled=bool(os.getenv('OPENAI_API_KEY'))
        ),
        ProviderConfig(
            provider=LLMProvider.GROQ,
            api_key=os.getenv('GROQ_API_KEY'),
            models=['mixtral-8x7b-32768', 'llama2-70b-4096'],
            default_model='mixtral-8x7b-32768',
            priority=2,
            rate_limit_per_minute=30,
            enabled=bool(os.getenv('GROQ_API_KEY'))
        ),
        ProviderConfig(
            provider=LLMProvider.GOOGLE,
            api_key=os.getenv('GOOGLE_API_KEY'),
            models=['gemini-pro'],
            default_model='gemini-pro',
            priority=3,
            rate_limit_per_minute=60,
            enabled=bool(os.getenv('GOOGLE_API_KEY'))
        ),
        ProviderConfig(
            provider=LLMProvider.OLLAMA,
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            models=['llama2', 'mistral'],
            default_model='llama2',
            priority=4,
            rate_limit_per_minute=120,  # Higher for local
            enabled=True  # Always enabled as fallback
        )
    ]
    
    return EnhancedLLMManager(provider_configs)