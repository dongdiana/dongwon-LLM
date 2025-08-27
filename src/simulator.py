"""
LLM Simulation module for persona-based Greek yogurt purchase decisions.
Uses LangChain and OpenAI GPT models for simulation.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import yaml
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from tqdm import asyncio as tqdm_asyncio

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonaSimulator:
    """
    Simulates persona-based purchase decisions using LLM.
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
        self.llm = self._initialize_llm()
        
        # Ensure results directory exists
        results_dir = Path(self.config["paths"]["results_dir"])
        results_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.simulation_stats = {
            "total_personas": 0,
            "successful_simulations": 0,
            "failed_simulations": 0,
            "purchase_decisions": {"0": 0, "1": 0},
            "start_time": None,
            "end_time": None
        }
    
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
    
    def _format_persona_prompt(self, persona: Dict[str, Any], market_context: str) -> Tuple[str, str]:
        """
        Format the user prompt with persona information.
        
        Args:
            persona: Persona dictionary with demographic info
            market_context: Greek yogurt market information
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Use the standard template with all required fields
        user_prompt = self.prompts["user_prompt_template"].format(
            age=persona["age"],
            gender=persona["gender"],
            region=persona["region"],
            education=persona["education"],
            occupation=persona["occupation"],
            household_size=persona["household_size"],
            market_context=market_context
        )
        
        return self.prompts["system_prompt"], user_prompt
    
    async def _simulate_single_persona(self, persona: Dict[str, Any], market_context: str) -> Dict[str, Any]:
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
            # Format prompts
            system_prompt, user_prompt = self._format_persona_prompt(persona, market_context)
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Call LLM
            start_time = time.time()
            response = await self.llm.ainvoke(messages)
            end_time = time.time()
            
            # Extract and validate response
            response_text = response.content.strip()
            purchase_decision = self._parse_purchase_decision(response_text)
            
            result = {
                "persona_id": persona_id,
                "persona": persona,
                "purchase_decision": purchase_decision,
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
            
            return {
                "persona_id": persona_id,
                "persona": persona,
                "purchase_decision": None,
                "raw_response": None,
                "response_time": None,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_purchase_decision(self, response: str) -> Optional[str]:
        """
        Parse the LLM response to extract purchase decision.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            "0" for no purchase, "1" for purchase, None if invalid
        """
        # Clean the response
        cleaned = response.strip().lower()
        
        # Direct matches
        if cleaned == "0":
            return "0"
        elif cleaned == "1":
            return "1"
        
        # Look for digits in the response
        for char in cleaned:
            if char in ["0", "1"]:
                return char
        
        # If no clear decision found
        logger.warning(f"Could not parse purchase decision from response: {response}")
        return None
    
    async def simulate_all_personas(self, personas: List[Dict[str, Any]], market_context: str) -> List[Dict[str, Any]]:
        """
        Simulate purchase decisions for all personas.
        
        Args:
            personas: List of persona dictionaries
            market_context: Market information string
            
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
    
    def save_results(self, results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        Save simulation results to JSON file.
        
        Args:
            results: List of simulation results
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.json"
        
        results_dir = Path(self.config["paths"]["results_dir"])
        file_path = results_dir / filename
        
        # Prepare output data
        output_data = {
            "metadata": {
                "simulation_time": datetime.now().isoformat(),
                "model_used": self.config["llm"]["model_name"],
                "total_personas": len(results),
                "statistics": self.simulation_stats
            },
            "results": results
        }
        
        # Save to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary report of simulation results.
        
        Args:
            results: List of simulation results
            
        Returns:
            Summary report dictionary
        """
        if not results:
            return {"error": "No results to summarize"}
        
        successful_results = [r for r in results if r["success"]]
        purchase_counts = {"0": 0, "1": 0, "invalid": 0}
        
        for result in successful_results:
            decision = result["purchase_decision"]
            if decision in ["0", "1"]:
                purchase_counts[decision] += 1
            else:
                purchase_counts["invalid"] += 1
        
        total_valid = purchase_counts["0"] + purchase_counts["1"]
        
        summary = {
            "total_personas": len(results),
            "successful_simulations": len(successful_results),
            "failed_simulations": len(results) - len(successful_results),
            "purchase_decisions": {
                "will_not_purchase": purchase_counts["0"],
                "will_purchase": purchase_counts["1"],
                "invalid_responses": purchase_counts["invalid"]
            },
            "purchase_rate": purchase_counts["1"] / total_valid if total_valid > 0 else 0,
            "statistics": self.simulation_stats
        }
        
        return summary
