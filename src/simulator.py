"""  
LLM Simulation module for persona-based product purchase decisions.
Uses LangChain and OpenAI GPT models for simulation.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import yaml
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from tqdm import asyncio as tqdm_asyncio

from .loader import ProductLoader

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonaSimulator:
    """
    Simulates persona-based product purchase decisions using LLM.
    """
    
    def __init__(self, config_path: str = "config.yaml", prompt_path: str = "prompt.yaml"):
        """
        Initialize the simulator with configuration and prompts.
        
        Args:
            config_path: Path to configuration YAML file
            prompt_path: Path to prompts YAML file
        """
        self.config = self._load_config(config_path)
        self.prompts = self._load_prompts(prompt_path)
        self._validate_prompt_config()
        
        # Initialize product loader
        self.product_loader = ProductLoader(self.config["paths"]["product_data_dir"])
        
        # Load product data
        self.product_info = self.product_loader.load_product_data(self.config["product"]["filename"])
        
        # Load Naver trend data once during initialization
        self.naver_trend_data = self.product_loader.load_naver_trend_data(self.config["product"]["filename"])
        logger.info(f"Loaded trend data for product: {self.config['product']['filename']}")
        
        self.llm = self._initialize_llm()
        

        
        # Statistics tracking
        self.simulation_stats = {
            "total_personas": 0,
            "successful_simulations": 0,
            "failed_simulations": 0,
            "purchase_decisions": {"0": 0, "1": 0},
            "start_time": None,
            "end_time": None
        }

        # Rate limiting tracking
        self.request_counter = 0

    def _apply_rate_limiting(self):
        """
        Apply rate limiting based on request count.
        - Wait 5 seconds after every 10 requests
        - Wait 60 seconds after every 100 requests
        """
        self.request_counter += 1

        # Check for 100-request interval
        if self.request_counter % 100 == 0:
            wait_time = 60  # 1 minute
            logger.info(f"Rate limit: Completed {self.request_counter} requests. Taking {wait_time}s break...")
            time.sleep(wait_time)
            logger.info("Rate limit break completed. Resuming simulation...")

        # Check for 10-request interval (but not also divisible by 100)
        elif self.request_counter % 10 == 0:
            wait_time = 5  # 5 seconds
            logger.info(f"Rate limit: Completed {self.request_counter} requests. Taking {wait_time}s break...")
            time.sleep(wait_time)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _load_prompts(self, prompt_path: str) -> Dict[str, str]:
        """Load prompts from YAML file."""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
                logger.info(f"Loaded prompts from {prompt_path}")
                return prompts
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise
    
    def _validate_prompt_config(self) -> None:
        """Validate that the configured prompts exist in the prompt file."""
        system_prompt_key = self.config["prompts"]["system_prompt"]
        user_prompt_key = self.config["prompts"]["user_prompt"]
        
        # Check if system prompt exists
        if system_prompt_key not in self.prompts:
            raise ValueError(f"System prompt '{system_prompt_key}' not found in prompt.yaml. "
                           f"Available prompts: {list(self.prompts.keys())}")
        
        # Check if user prompt exists
        if user_prompt_key not in self.prompts:
            raise ValueError(f"User prompt '{user_prompt_key}' not found in prompt.yaml. "
                           f"Available prompts: {list(self.prompts.keys())}")
        
        logger.info(f"Using system prompt: {system_prompt_key}")
        logger.info(f"Using user prompt: {user_prompt_key}")
    

    
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the LangChain LLM."""
        llm_config = self.config["llm"]
        
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        llm = ChatOpenAI(
            model=llm_config["model_name"],
            temperature=llm_config["temperature"],
            request_timeout=llm_config["timeout"],
            openai_api_key=api_key,
            max_retries=self.config["openai"]["max_retries"]
        )
        
        logger.info(f"Initialized LLM: {llm_config['model_name']}")
        return llm
    
    def _format_persona_prompt(self, persona: Dict[str, Any], market_context: str, search_summary: str) -> Tuple[str, str]:
        """
        Format the prompts with persona information, market context, and search summary.
        
        Args:
            persona: Persona dictionary with demographic info
            market_context: Product market information
            search_summary: Search trend analysis data
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Extract product name from product info
        product_name = self.get_product_name()
        
        # Get prompt keys from config
        system_prompt_key = self.config["prompts"]["system_prompt"]
        user_prompt_key = self.config["prompts"]["user_prompt"]
        
        # Format system prompt with all required variables
        system_prompt = self.prompts[system_prompt_key].format(
            age=persona["age"],
            gender=persona["gender"],
            income=persona["income"],
            household_size=persona["household_size"],
            product_name=product_name,
            market_context=market_context,
            search_summary=search_summary
        )
        
        # Format user prompt with product name
        user_prompt = self.prompts[user_prompt_key].format(
            product_name=product_name
        )
        
        return system_prompt, user_prompt
    
    def get_product_name(self) -> str:
        """Get the product name from config filename."""
        # Use filename from config directly as the product name
        filename = self.config["product"]["filename"]
        # Remove .json extension if present
        if filename.endswith('.json'):
            return filename[:-5]
        return filename
    
    def _get_market_context(self) -> str:
        """Extract market context from product info only (excludes search/trend data)."""
        return self.product_loader.format_market_context(self.product_info)
    
    def _get_search_summary(self) -> str:
        """
        Extract Naver trend data and format it as search summary.
        Uses cached trend data loaded during initialization.
        
        Returns:
            Formatted string with search trend analysis
        """
        # Get product name for formatting
        product_name = self.get_product_name()
        
        # Format the cached data using the loader
        return self.product_loader.format_search_summary(self.naver_trend_data, product_name)
    
    async def _simulate_single_persona(self, persona: Dict[str, Any], market_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Simulate purchase decision for a single persona.
        
        Args:
            persona: Persona dictionary
            market_context: Market information string
            
        Returns:
            Dictionary with simulation results
        """
        persona_id = persona.get("id", "unknown")
        
        try:
            # Use provided market context or extract from product info
            if market_context is None:
                market_context = self._get_market_context()
            
            # Get search summary separately
            search_summary = self._get_search_summary()
            
            # Format prompts with all required data
            system_prompt, user_prompt = self._format_persona_prompt(persona, market_context, search_summary)
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Call LLM
            start_time = time.time()
            response = await self.llm.ainvoke(messages)
            end_time = time.time()

            # Apply rate limiting after successful API call
            self._apply_rate_limiting()

            # Extract and validate response
            response_text = response.content.strip()
            purchase_decision, reasoning = self._parse_purchase_decision(response_text)
            
            result = {
                "persona_id": persona_id,
                "persona": persona,
                "purchase_decision": purchase_decision,
                "reasoning": reasoning,
                "raw_response": response_text,
                "response_time": end_time - start_time,
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update statistics
            self.simulation_stats["successful_simulations"] += 1
            if purchase_decision in ["0", "1"]:
                self.simulation_stats["purchase_decisions"][purchase_decision] += 1
            
            logger.debug(f"Persona {persona_id}: Decision = {purchase_decision}")
            return result
            
        except Exception as e:
            logger.error(f"Error simulating persona {persona_id}: {e}")
            self.simulation_stats["failed_simulations"] += 1

            # Apply rate limiting even on failed requests to maintain consistent pacing
            self._apply_rate_limiting()

            return {
                "persona_id": persona_id,
                "persona": persona,
                "purchase_decision": None,
                "reasoning": None,
                "raw_response": None,
                "response_time": None,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_purchase_decision(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the LLM response to extract purchase decision and reasoning.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Tuple of (decision, reasoning) where decision is "0"/"1" or None if invalid
        """
        # Clean the response
        cleaned = response.strip()
        
        # Try to match pattern: "number, reasoning" or "number reasoning"
        patterns = [
            r'^([01])\s*,\s*(.+)$',  # "1, reasoning"
            r'^([01])\s+(.+)$',      # "1 reasoning"
            r'^([01])\s*(.+)$'       # "1reasoning"
        ]
        
        for pattern in patterns:
            match = re.match(pattern, cleaned, re.DOTALL)
            if match:
                decision = match.group(1)
                reasoning = match.group(2).strip()
                return decision, reasoning
        
        # If structured format not found, try to find just the number
        for char in cleaned:
            if char in ["0", "1"]:
                # Extract reasoning as everything after the number
                idx = cleaned.find(char)
                reasoning = cleaned[idx+1:].strip().lstrip(',').strip()
                return char, reasoning if reasoning else None
        
        # If no clear decision found
        logger.warning(f"Could not parse purchase decision from response: {response}")
        return None, None
    
    async def simulate_all_personas(self, personas: List[Dict[str, Any]], market_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Simulate purchase decisions for all personas.
        
        Args:
            personas: List of persona dictionaries
            market_context: Optional market information string (defaults to product info)
            
        Returns:
            List of simulation results
        """
        if not personas:
            logger.warning("No personas provided for simulation")
            return []
        
        self.simulation_stats["total_personas"] = len(personas)
        self.simulation_stats["start_time"] = datetime.now().isoformat()
        
        logger.info(f"Starting simulation for {len(personas)} personas")
        
        # Create semaphore to limit concurrent requests
        batch_size = self.config["simulation"]["batch_size"]
        semaphore = asyncio.Semaphore(batch_size)
        
        async def bounded_simulate(persona):
            async with semaphore:
                return await self._simulate_single_persona(persona, market_context)
        
        # Run simulations with progress bar
        tasks = [bounded_simulate(persona) for persona in personas]
        results = []
        
        # Use tqdm for progress tracking
        for task in tqdm_asyncio.tqdm.as_completed(tasks, desc="Simulating personas"):
            result = await task
            results.append(result)
        
        self.simulation_stats["end_time"] = datetime.now().isoformat()
        logger.info(f"Simulation completed. Success: {self.simulation_stats['successful_simulations']}, "
                   f"Failed: {self.simulation_stats['failed_simulations']}")
        
        return results
    

