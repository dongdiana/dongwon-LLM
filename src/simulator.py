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
from langchain.schema import HumanMessage, SystemMessage, AIMessage
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
        
        # Load product data based on prompt type
        prompt_type = self.config["prompts"]["type"]
        if prompt_type == "D":
            # Load TypeD product data
            self.product_info = self.product_loader.load_type_d_product_data(self.config["product"]["filename"])
            self.type_d_data = self.product_info  # Store TypeD data separately for easy access
            logger.info(f"Loaded TypeD product data for: {self.config['product']['filename']}")
        else:
            # Load regular product data
            self.product_info = self.product_loader.load_product_data(self.config["product"]["filename"])
            self.type_d_data = None
        
        # Load Naver trend data once during initialization (not used for TypeD)
        if prompt_type != "D":
            self.naver_trend_data = self.product_loader.load_naver_trend_data(self.config["product"]["filename"])
            logger.info(f"Loaded trend data for product: {self.config['product']['filename']}")
        else:
            self.naver_trend_data = {}
        
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
            wait_time = 10
            logger.info(f"Rate limit: Completed {self.request_counter} requests. Taking {wait_time}s break...")
            time.sleep(wait_time)
            logger.info("Rate limit break completed. Resuming simulation...")

        # Check for 10-request interval (but not also divisible by 100)
        elif self.request_counter % 20 == 0:
            wait_time = 3  # 3 seconds
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
        """Validate that the configured prompt type exists in the prompt file."""
        prompt_type = self.config["prompts"]["type"]
        
        if prompt_type not in ["A", "B", "C", "D"]:
            raise ValueError(f"Prompt type must be 'A', 'B', 'C', or 'D', got: {prompt_type}")
        
        # Check if required prompts exist based on type
        if prompt_type == "A":
            required_prompts = ["system_prompt_A", "user_prompt_A"]
        elif prompt_type == "B":
            required_prompts = ["system_prompt_B"]
            # Find all user_prompt_B* prompts
            b_user_prompts = [key for key in self.prompts.keys() if key.startswith("user_prompt_B")]
            required_prompts.extend(b_user_prompts)
        elif prompt_type == "C":
            required_prompts = ["system_prompt_C", "user_prompt_C"]
        else:  # type D
            required_prompts = ["system_prompt_D", "user_prompt_D"]
        
        for prompt_key in required_prompts:
            if prompt_key not in self.prompts:
                raise ValueError(f"Required prompt '{prompt_key}' not found in prompt.yaml for type {prompt_type}. "
                               f"Available prompts: {list(self.prompts.keys())}")
        
        logger.info(f"Using prompt type: {prompt_type}")
        if prompt_type == "B":
            b_prompts = [key for key in required_prompts if key.startswith("user_prompt_B")]
            logger.info(f"Found {len(b_prompts)} user prompts for type B: {b_prompts}")
    

    
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
    
    def _format_persona_prompt_type_a(self, persona: Dict[str, Any], market_context: str, search_summary: str) -> Tuple[str, str]:
        """
        Format the prompts for Type A (single question) with persona information.
        
        Args:
            persona: Persona dictionary with demographic info
            market_context: Product market information
            search_summary: Search trend analysis data
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Extract product name from product info
        product_name = self.get_product_name()
        
        # Format system prompt with all required variables
        system_prompt = self.prompts["system_prompt_A"].format(
            age=persona["age"],
            gender=persona["gender"],
            income=persona["income"],
            household_size=persona["household_size"],
            product_name=product_name,
            market_context=market_context,
            search_summary=search_summary
        )
        
        # Format user prompt with product name
        user_prompt = self.prompts["user_prompt_A"].format(
            product_name=product_name
        )
        
        return system_prompt, user_prompt
    
    def _format_persona_prompt_type_b(self, persona: Dict[str, Any], user_prompt_key: str, product_options: str = "", quantity_context: str = "") -> Tuple[str, str]:
        """
        Format the prompts for Type B (multi-question) with detailed persona information.
        
        Args:
            persona: Detailed persona dictionary with reasoning and persona_info
            user_prompt_key: Key for the specific user prompt (e.g., "user_prompt_B1")
            product_options: Formatted product options string (for B1)
            quantity_context: Context from previous question (for B2)
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Get product information for formatting
        product_name = self.get_product_name()
        market_context = self._get_market_context()
        search_summary = self._get_search_summary()
        
        # Format system prompt with detailed persona data and market information
        system_prompt = self.prompts["system_prompt_B"].format(
            reasoning=persona["reasoning"],
            persona_info=persona["persona_info"],
            product_name=product_name,
            market_context=market_context,
            search_summary=search_summary
        )
        
        # Format user prompt based on which question this is
        if user_prompt_key == "user_prompt_B1":
            user_prompt = self.prompts[user_prompt_key].format(
                product_name=product_name,
                product_options=product_options
            )
        elif user_prompt_key == "user_prompt_B2":
            # Use the context from the previous question if needed
            user_prompt = self.prompts[user_prompt_key]
        else:
            # Generic formatting for any additional B prompts
            user_prompt = self.prompts[user_prompt_key]
        
        return system_prompt, user_prompt
    
    def _format_persona_prompt_type_c(self, persona: Dict[str, Any], target_product: str, current_product: str) -> Tuple[str, str]:
        """
        Format the prompts for Type C (product comparison) with detailed persona information.
        
        Args:
            persona: Detailed persona dictionary with reasoning and persona_info
            target_product: Formatted target product information
            current_product: Formatted current product information
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Get product information for formatting
        product_name = self.get_product_name()
        market_context = self._get_market_context()
        search_summary = self._get_search_summary()
        
        # Format system prompt with detailed persona data and market information
        system_prompt = self.prompts["system_prompt_C"].format(
            reasoning=persona["reasoning"],
            persona_info=persona["persona_info"],
            product_name=product_name,
            market_context=market_context,
            search_summary=search_summary
        )
        
        # Format user prompt with product comparison information
        user_prompt = self.prompts["user_prompt_C"].format(
            product_name=product_name,
            target_product=target_product,
            current_product=current_product
        )
        
        return system_prompt, user_prompt
    
    def _format_persona_prompt_type_d(self, persona: Dict[str, Any]) -> Tuple[str, str]:
        """
        Format the prompts for Type D with detailed persona information and TypeD product data.
        
        Args:
            persona: Detailed persona dictionary with reasoning and persona_info
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Get product information for TypeD
        product_name = self.get_product_name()
        product_info = self.product_loader.get_type_d_product_info(self.type_d_data)
        product_options = self.product_loader.get_type_d_product_options(self.type_d_data)
        
        # Format system prompt with detailed persona data and product information
        system_prompt = self.prompts["system_prompt_D"].format(
            reasoning=persona["reasoning"],
            persona_info=persona["persona_info"],
            product_name=product_name,
            product_info=product_info
        )
        
        # Format user prompt with product options
        user_prompt = self.prompts["user_prompt_D"].format(
            product_name=product_name,
            product_options=product_options
        )
        
        return system_prompt, user_prompt
    
    def get_product_name(self) -> str:
        """Get the product name from config filename or TypeD data."""
        prompt_type = self.config["prompts"]["type"]
        
        if prompt_type == "D" and self.type_d_data:
            # For TypeD, get product name from top-level schema in TypeD data
            return self.product_loader.get_type_d_product_name(self.type_d_data)
        else:
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
        Handles both Type A (single question) and Type B (multi-question) simulations.
        
        Args:
            persona: Persona dictionary
            market_context: Market information string (only used for Type A)
            
        Returns:
            Dictionary with simulation results
        """
        prompt_type = self.config["prompts"]["type"]
        
        if prompt_type == "A":
            return await self._simulate_single_persona_type_a(persona, market_context)
        elif prompt_type == "B":
            return await self._simulate_single_persona_type_b(persona)
        elif prompt_type == "C":
            return await self._simulate_single_persona_type_c(persona)
        else:  # Type D
            return await self._simulate_single_persona_type_d(persona)
    
    async def _simulate_single_persona_type_a(self, persona: Dict[str, Any], market_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Simulate Type A (single question) for a single persona.
        
        Args:
            persona: Persona dictionary with demographic info
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
            system_prompt, user_prompt = self._format_persona_prompt_type_a(persona, market_context, search_summary)
            
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
    
    async def _simulate_single_persona_type_b(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate Type B (multi-question) for a single persona.
        
        Args:
            persona: Detailed persona dictionary
            
        Returns:
            Dictionary with simulation results
        """
        persona_id = persona.get("uuid", "unknown")
        
        try:
            # Generate randomized product options
            product_options, product_names = self.product_loader.generate_product_options(self.product_info)
            
            # Get available user prompts for Type B
            b_user_prompts = sorted([key for key in self.prompts.keys() if key.startswith("user_prompt_B")])
            
            all_responses = []
            total_response_time = 0
            selected_product = None
            selected_quantity = None
            
            # Initialize conversation history with system prompt
            system_prompt, _ = self._format_persona_prompt_type_b(persona, "user_prompt_B1")
            conversation_history = [SystemMessage(content=system_prompt)]
            
            # Question 1: Product selection
            if "user_prompt_B1" in b_user_prompts:
                _, user_prompt = self._format_persona_prompt_type_b(
                    persona, "user_prompt_B1", product_options=product_options
                )
                
                # Add the user question to conversation history
                conversation_history.append(HumanMessage(content=user_prompt))
                
                start_time = time.time()
                response = await self.llm.ainvoke(conversation_history)
                end_time = time.time()
                total_response_time += (end_time - start_time)
                
                self._apply_rate_limiting()
                
                response_text = response.content.strip()
                selected_number, selected_product, reasoning1 = self._parse_product_selection(response_text, product_names)
                
                # Add the assistant response to conversation history
                conversation_history.append(AIMessage(content=response_text))
                
                all_responses.append({
                    "question": "user_prompt_B1",
                    "response": response_text,
                    "selected_number": selected_number,
                    "selected_product": selected_product,
                    "reasoning": reasoning1
                })
            
            # Question 2: Quantity selection (if product was selected)
            if "user_prompt_B2" in b_user_prompts and selected_product:
                _, user_prompt = self._format_persona_prompt_type_b(
                    persona, "user_prompt_B2"
                )
                
                # Add the user question to conversation history (maintains previous context)
                conversation_history.append(HumanMessage(content=user_prompt))
                
                start_time = time.time()
                response = await self.llm.ainvoke(conversation_history)
                end_time = time.time()
                total_response_time += (end_time - start_time)
                
                self._apply_rate_limiting()
                
                response_text = response.content.strip()
                selected_quantity, reasoning2 = self._parse_quantity_selection(response_text)
                
                # Add the assistant response to conversation history
                conversation_history.append(AIMessage(content=response_text))
                
                all_responses.append({
                    "question": "user_prompt_B2",
                    "response": response_text,
                    "selected_quantity": selected_quantity,
                    "reasoning": reasoning2
                })
            
            result = {
                "persona_id": persona_id,
                "persona": persona,
                "product_options_order": product_names,
                "selected_product": selected_product,
                "selected_quantity": selected_quantity,
                "all_responses": all_responses,
                "response_time": total_response_time,
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update statistics
            self.simulation_stats["successful_simulations"] += 1
            
            logger.debug(f"Persona {persona_id}: Selected product = {selected_product}, Quantity = {selected_quantity}")
            return result
            
        except Exception as e:
            logger.error(f"Error simulating persona {persona_id}: {e}")
            self.simulation_stats["failed_simulations"] += 1

            # Apply rate limiting even on failed requests to maintain consistent pacing
            self._apply_rate_limiting()

            return {
                "persona_id": persona_id,
                "persona": persona,
                "product_options_order": [],
                "selected_product": None,
                "selected_quantity": None,
                "all_responses": [],
                "response_time": None,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _simulate_single_persona_type_c(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate Type C (product comparison/conversion) for a single persona.
        
        Args:
            persona: Detailed persona dictionary
            
        Returns:
            Dictionary with simulation results
        """
        persona_id = persona.get("uuid", "unknown")
        
        try:
            # Get current product from persona
            current_product_name = persona.get("raw_data", {}).get("기존사용제품", "")
            if not current_product_name:
                # Fallback to direct access if raw_data doesn't exist
                current_product_name = persona.get("기존사용제품", "")
            
            if not current_product_name:
                raise ValueError("No current product found in persona data")
            
            # Get target and current product information
            target_product = self.product_loader.get_target_product_info(self.product_info)
            current_product = self.product_loader.get_current_product_info(self.product_info, current_product_name)
            
            # Format prompts with product comparison data
            system_prompt, user_prompt = self._format_persona_prompt_type_c(
                persona, target_product, current_product
            )
            
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
            conversion_decision, reasoning = self._parse_purchase_decision(response_text)
            
            result = {
                "persona_id": persona_id,
                "persona": persona,
                "current_product": current_product_name,
                "target_product_info": target_product,
                "current_product_info": current_product,
                "conversion_decision": conversion_decision,
                "reasoning": reasoning,
                "raw_response": response_text,
                "response_time": end_time - start_time,
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update statistics
            self.simulation_stats["successful_simulations"] += 1
            if conversion_decision in ["0", "1"]:
                self.simulation_stats["purchase_decisions"][conversion_decision] += 1
            
            logger.debug(f"Persona {persona_id}: Conversion decision = {conversion_decision}, Current product = {current_product_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error simulating persona {persona_id}: {e}")
            self.simulation_stats["failed_simulations"] += 1

            # Apply rate limiting even on failed requests to maintain consistent pacing
            self._apply_rate_limiting()

            return {
                "persona_id": persona_id,
                "persona": persona,
                "current_product": persona.get("raw_data", {}).get("기존사용제품", persona.get("기존사용제품", "Unknown")),
                "target_product_info": None,
                "current_product_info": None,
                "conversion_decision": None,
                "reasoning": None,
                "raw_response": None,
                "response_time": None,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _simulate_single_persona_type_d(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate Type D (TypeD product selection) for a single persona.
        
        Args:
            persona: Detailed persona dictionary
            
        Returns:
            Dictionary with simulation results
        """
        persona_id = persona.get("uuid", "unknown")
        
        try:
            # Format prompts with TypeD product data
            system_prompt, user_prompt = self._format_persona_prompt_type_d(persona)
            
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

            # Extract and validate response for TypeD (category selection)
            response_text = response.content.strip()
            selection_number, selected_category, reasoning = self._parse_type_d_selection(response_text)
            
            result = {
                "persona_id": persona_id,
                "persona": persona,
                "product_name": self.get_product_name(),
                "selection_number": selection_number,
                "selected_category": selected_category,
                "reasoning": reasoning,
                "raw_response": response_text,
                "response_time": end_time - start_time,
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update statistics
            self.simulation_stats["successful_simulations"] += 1
            # For TypeD, we don't use the purchase_decisions tracking
            
            logger.debug(f"Persona {persona_id}: Selected category = {selected_category} (option {selection_number})")
            return result
            
        except Exception as e:
            logger.error(f"Error simulating persona {persona_id}: {e}")
            self.simulation_stats["failed_simulations"] += 1

            # Apply rate limiting even on failed requests to maintain consistent pacing
            self._apply_rate_limiting()

            return {
                "persona_id": persona_id,
                "persona": persona,
                "product_name": self.get_product_name() if self.type_d_data else "Unknown",
                "selection_number": None,
                "selected_category": None,
                "reasoning": None,
                "raw_response": None,
                "response_time": None,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_purchase_decision(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the LLM response to extract purchase decision and reasoning (Type A).
        
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
    
    def _parse_product_selection(self, response: str, product_names: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse the LLM response to extract product selection and reasoning (Type B1).
        
        Args:
            response: Raw LLM response text
            product_names: List of product names in order of options
            
        Returns:
            Tuple of (selected_number, selected_product_name, reasoning)
        """
        # Clean the response
        cleaned = response.strip()
        
        # Try to match pattern: "number, reasoning" or "number reasoning"
        patterns = [
            r'^(\d+)\s*,\s*(.+)$',  # "1, reasoning"
            r'^(\d+)\s+(.+)$',      # "1 reasoning"
            r'^(\d+)\s*(.+)$'       # "1reasoning"
        ]
        
        for pattern in patterns:
            match = re.match(pattern, cleaned, re.DOTALL)
            if match:
                number_str = match.group(1)
                reasoning = match.group(2).strip()
                
                try:
                    number = int(number_str)
                    if 1 <= number <= len(product_names):
                        product_name = product_names[number - 1]  # Convert to 0-based index
                        return number_str, product_name, reasoning
                    else:
                        logger.warning(f"Selected number {number} is out of range (1-{len(product_names)})")
                        return number_str, None, reasoning
                except ValueError:
                    logger.warning(f"Could not convert '{number_str}' to integer")
                    return number_str, None, reasoning
        
        # If structured format not found, try to find just the number
        for char in cleaned:
            if char.isdigit():
                try:
                    number = int(char)
                    if 1 <= number <= len(product_names):
                        product_name = product_names[number - 1]
                        # Extract reasoning as everything after the number
                        idx = cleaned.find(char)
                        reasoning = cleaned[idx+1:].strip().lstrip(',').strip()
                        return char, product_name, reasoning if reasoning else None
                except ValueError:
                    continue
        
        # If no clear selection found
        logger.warning(f"Could not parse product selection from response: {response}")
        return None, None, None
    
    def _parse_quantity_selection(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the LLM response to extract quantity selection and reasoning (Type B2).
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Tuple of (quantity, reasoning)
        """
        # Clean the response
        cleaned = response.strip()
        
        # Try to match pattern: "number, reasoning" or "number reasoning"
        patterns = [
            r'^(\d+)\s*,\s*(.+)$',  # "2, reasoning"
            r'^(\d+)\s+(.+)$',      # "2 reasoning"
            r'^(\d+)\s*(.+)$'       # "2reasoning"
        ]
        
        for pattern in patterns:
            match = re.match(pattern, cleaned, re.DOTALL)
            if match:
                quantity = match.group(1)
                reasoning = match.group(2).strip()
                return quantity, reasoning
        
        # If structured format not found, try to find just the number
        for i, char in enumerate(cleaned):
            if char.isdigit():
                # Try to get the full number (could be multi-digit)
                num_start = i
                num_end = i + 1
                while num_end < len(cleaned) and cleaned[num_end].isdigit():
                    num_end += 1
                
                quantity = cleaned[num_start:num_end]
                reasoning = cleaned[num_end:].strip().lstrip(',').strip()
                return quantity, reasoning if reasoning else None
        
        # If no clear quantity found
        logger.warning(f"Could not parse quantity selection from response: {response}")
        return None, None
    
    def _parse_type_d_selection(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse the LLM response to extract TypeD category selection and reasoning.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Tuple of (selection_number, selected_category, reasoning)
        """
        # Clean the response
        cleaned = response.strip()
        
        # Get category options from TypeD data
        if not self.type_d_data:
            logger.warning("No TypeD data available for parsing")
            return None, None, None
        
        categories = []
        product_name = self.product_loader.get_type_d_product_name(self.type_d_data)
        if product_name in self.type_d_data and "category" in self.type_d_data[product_name]:
            categories = self.type_d_data[product_name]["category"]
        
        if not categories:
            logger.warning("No categories found in TypeD data")
            return None, None, None
        
        # Try to match pattern: "number, reasoning" or "number reasoning"
        patterns = [
            r'^(\d+)\s*,\s*(.+)$',  # "1, reasoning"
            r'^(\d+)\s+(.+)$',      # "1 reasoning"
            r'^(\d+)\s*(.+)$'       # "1reasoning"
        ]
        
        for pattern in patterns:
            match = re.match(pattern, cleaned, re.DOTALL)
            if match:
                number_str = match.group(1)
                reasoning = match.group(2).strip()
                
                try:
                    number = int(number_str)
                    if 1 <= number <= len(categories):
                        selected_category = categories[number - 1]  # Convert to 0-based index
                        return number_str, selected_category, reasoning
                    else:
                        logger.warning(f"Selected number {number} is out of range (1-{len(categories)})")
                        return number_str, None, reasoning
                except ValueError:
                    logger.warning(f"Could not convert '{number_str}' to integer")
                    return number_str, None, reasoning
        
        # If structured format not found, try to find just the number
        for char in cleaned:
            if char.isdigit():
                try:
                    number = int(char)
                    if 1 <= number <= len(categories):
                        selected_category = categories[number - 1]
                        # Extract reasoning as everything after the number
                        idx = cleaned.find(char)
                        reasoning = cleaned[idx+1:].strip().lstrip(',').strip()
                        return char, selected_category, reasoning if reasoning else None
                except ValueError:
                    continue
        
        # If no clear selection found
        logger.warning(f"Could not parse TypeD selection from response: {response}")
        return None, None, None
    
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
    

