"""
Prompt Registry for managing Jinja2 templates in RAG system.
Provides centralized prompt management and template rendering.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PromptType(str, Enum):
    """Types of prompts available."""
    PRODUCT_RECOMMENDATION = "product_recommendation"
    PRODUCT_COMPARISON = "product_comparison"
    PRODUCT_INFO = "product_info"
    REVIEW_SUMMARY = "review_summary"
    TROUBLESHOOTING = "troubleshooting"
    GENERAL_QUERY = "general_query"


@dataclass
class PromptTemplate:
    """Represents a prompt template."""
    name: str
    template_path: str
    macro_name: str
    description: str
    version: str = "1.0.0"
    parameters: List[str] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []


class PromptRegistry:
    """Registry for managing prompt templates."""
    
    def __init__(self, templates_dir: str = None, config_file: str = None):
        """Initialize prompt registry."""
        
        # Set default paths
        self.base_dir = Path(__file__).parent
        self.templates_dir = Path(templates_dir) if templates_dir else self.base_dir / "templates"
        self.config_file = Path(config_file) if config_file else self.base_dir / "config.yaml"
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Registry of templates
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Load configuration
        self._load_config()
        
        # Load templates
        self._load_templates()
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    self._process_config(config)
                    logger.info(f"Loaded prompt configuration from {self.config_file}")
            else:
                logger.warning(f"Config file not found: {self.config_file}")
                self._create_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration."""
        default_config = {
            "templates": {
                "product_recommendation": {
                    "name": "Product Recommendation",
                    "template_path": "rag_templates.j2",
                    "macro_name": "product_recommendation_prompt",
                    "description": "Template for product recommendation responses",
                    "parameters": ["query", "products", "reviews", "search_context"]
                },
                "product_comparison": {
                    "name": "Product Comparison",
                    "template_path": "rag_templates.j2",
                    "macro_name": "product_comparison_prompt",
                    "description": "Template for product comparison responses",
                    "parameters": ["query", "products", "reviews", "search_context"]
                },
                "product_info": {
                    "name": "Product Information",
                    "template_path": "rag_templates.j2",
                    "macro_name": "product_info_prompt",
                    "description": "Template for product information responses",
                    "parameters": ["query", "products", "reviews", "search_context"]
                },
                "review_summary": {
                    "name": "Review Summary",
                    "template_path": "rag_templates.j2",
                    "macro_name": "review_summary_prompt",
                    "description": "Template for review summary responses",
                    "parameters": ["query", "products", "reviews", "search_context"]
                },
                "troubleshooting": {
                    "name": "Troubleshooting",
                    "template_path": "rag_templates.j2",
                    "macro_name": "troubleshooting_prompt",
                    "description": "Template for troubleshooting responses",
                    "parameters": ["query", "products", "reviews", "search_context"]
                },
                "general_query": {
                    "name": "General Query",
                    "template_path": "rag_templates.j2",
                    "macro_name": "general_query_prompt",
                    "description": "Template for general query responses",
                    "parameters": ["query", "products", "reviews", "search_context"]
                }
            }
        }
        
        # Save default config
        try:
            os.makedirs(self.config_file.parent, exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Created default configuration at {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to create default configuration: {e}")
    
    def _process_config(self, config: Dict[str, Any]):
        """Process configuration and create template objects."""
        templates_config = config.get("templates", {})
        
        for template_id, template_config in templates_config.items():
            template = PromptTemplate(
                name=template_config.get("name", template_id),
                template_path=template_config.get("template_path", "rag_templates.j2"),
                macro_name=template_config.get("macro_name", f"{template_id}_prompt"),
                description=template_config.get("description", ""),
                version=template_config.get("version", "1.0.0"),
                parameters=template_config.get("parameters", [])
            )
            self.templates[template_id] = template
    
    def _load_templates(self):
        """Load and validate templates."""
        for template_id, template in self.templates.items():
            try:
                # Load template file
                template_file = self.env.get_template(template.template_path)
                
                # Check if macro exists
                if hasattr(template_file.module, template.macro_name):
                    logger.info(f"Loaded template: {template_id}")
                else:
                    logger.warning(f"Macro {template.macro_name} not found in {template.template_path}")
                    
            except Exception as e:
                logger.error(f"Failed to load template {template_id}: {e}")
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[str]:
        """List all available template IDs."""
        return list(self.templates.keys())
    
    def render_prompt(self, template_id: str, **kwargs) -> str:
        """Render a prompt using the specified template."""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        try:
            # Get the template file
            template_file = self.env.get_template(template.template_path)
            
            # Get the macro
            macro = getattr(template_file.module, template.macro_name)
            
            # Render the macro with parameters
            rendered = macro(**kwargs)
            
            return rendered
            
        except Exception as e:
            logger.error(f"Failed to render template {template_id}: {e}")
            raise
    
    def render_rag_prompt(self, prompt_type: PromptType, 
                         query: str, products: List[Dict], 
                         reviews: List[Dict], search_context: Dict) -> str:
        """Render RAG prompt using the specified type."""
        
        # Map prompt type to template ID
        template_id = prompt_type.value
        
        # Prepare parameters
        params = {
            "query": query,
            "products": products,
            "reviews": reviews,
            "search_context": search_context
        }
        
        return self.render_prompt(template_id, **params)
    
    def validate_template(self, template_id: str) -> bool:
        """Validate that a template can be loaded and rendered."""
        template = self.get_template(template_id)
        if not template:
            return False
        
        try:
            # Test render with minimal parameters
            test_params = {param: "" for param in template.parameters}
            test_params.update({
                "query": "test query",
                "products": [],
                "reviews": [],
                "search_context": {
                    "query_type": "test",
                    "search_strategy": "test",
                    "total_results": 0,
                    "reranking_applied": False
                }
            })
            
            self.render_prompt(template_id, **test_params)
            return True
            
        except Exception as e:
            logger.error(f"Template validation failed for {template_id}: {e}")
            return False
    
    def get_template_info(self, template_id: str) -> Dict[str, Any]:
        """Get information about a template."""
        template = self.get_template(template_id)
        if not template:
            return {}
        
        return {
            "name": template.name,
            "description": template.description,
            "version": template.version,
            "parameters": template.parameters,
            "template_path": template.template_path,
            "macro_name": template.macro_name
        }
    
    def add_template(self, template_id: str, template: PromptTemplate):
        """Add a new template to the registry."""
        self.templates[template_id] = template
        logger.info(f"Added template: {template_id}")
    
    def remove_template(self, template_id: str):
        """Remove a template from the registry."""
        if template_id in self.templates:
            del self.templates[template_id]
            logger.info(f"Removed template: {template_id}")
    
    def reload_templates(self):
        """Reload all templates from configuration."""
        self.templates.clear()
        self._load_config()
        self._load_templates()
        logger.info("Reloaded all templates")
    
    def export_config(self, filepath: str):
        """Export current configuration to file."""
        config = {
            "templates": {
                template_id: {
                    "name": template.name,
                    "template_path": template.template_path,
                    "macro_name": template.macro_name,
                    "description": template.description,
                    "version": template.version,
                    "parameters": template.parameters
                }
                for template_id, template in self.templates.items()
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Exported configuration to {filepath}")


# Global registry instance
_registry: Optional[PromptRegistry] = None


def get_registry() -> PromptRegistry:
    """Get the global prompt registry instance."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry


def render_rag_prompt(prompt_type: PromptType, query: str, 
                     products: List[Dict], reviews: List[Dict], 
                     search_context: Dict) -> str:
    """Convenience function to render RAG prompt."""
    registry = get_registry()
    return registry.render_rag_prompt(prompt_type, query, products, reviews, search_context)


def list_available_templates() -> List[str]:
    """List all available template IDs."""
    registry = get_registry()
    return registry.list_templates()


def validate_all_templates() -> Dict[str, bool]:
    """Validate all templates in the registry."""
    registry = get_registry()
    return {template_id: registry.validate_template(template_id) 
            for template_id in registry.list_templates()} 