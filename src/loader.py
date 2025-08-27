"""
Data loader module for persona and product information.
Handles loading JSONL persona files and product JSON data.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonaLoader:
    """Loads persona data from CSV file."""
    
    def __init__(self, persona_csv_path: str):
        self.persona_csv_path = Path(persona_csv_path)
        
    def load_all_personas(self) -> List[Dict[str, Any]]:
        """
        Load all personas from CSV file.
        
        Returns:
            List of persona dictionaries
        """
        personas = []
        
        if not self.persona_csv_path.exists():
            logger.error(f"Persona CSV file does not exist: {self.persona_csv_path}")
            return personas
            
        try:
            # Load CSV file
            df = pd.read_csv(self.persona_csv_path, encoding='utf-8')
            logger.info(f"Loaded CSV file with {len(df)} rows")
            
            # Convert to list of dictionaries
            for _, row in df.iterrows():
                persona = {
                    "id": str(row["id"]),
                    "region": row["지역"],
                    "gender": row["성별"],
                    "education": row["학력"],
                    "occupation": row["직업"],
                    "age": str(row["연령대"]),
                    "household_size": str(row["가구원수"])
                }
                
                # Validate persona
                if self._validate_persona(persona):
                    personas.append(persona)
                else:
                    logger.warning(f"Invalid persona with id {persona.get('id', 'unknown')}")
            
            logger.info(f"Successfully loaded {len(personas)} valid personas")
            return personas
            
        except Exception as e:
            logger.error(f"Error loading persona CSV file: {e}")
            return []
    

    
    def _validate_persona(self, persona: Dict[str, Any]) -> bool:
        """
        Validate that a persona has all required fields.
        
        Args:
            persona: Persona dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["id", "region", "gender", "age", "education", "occupation", "household_size"]
        
        # Check for all required fields
        for field in required_fields:
            if field not in persona or persona[field] is None or str(persona[field]).strip() == "":
                logger.warning(f"Missing or empty required field '{field}' in persona")
                return False
                
        return True


class ProductLoader:
    """Loads product information from JSON files."""
    
    def __init__(self, product_data_dir: str):
        self.product_data_dir = Path(product_data_dir)
        
    def load_greek_yogurt_data(self) -> Dict[str, Any]:
        """
        Load Greek yogurt market information.
        
        Returns:
            Dictionary containing Greek yogurt market data
        """
        greek_yogurt_file = self.product_data_dir / "그릭요거트.json"
        
        if not greek_yogurt_file.exists():
            logger.error(f"Greek yogurt data file not found: {greek_yogurt_file}")
            return {}
            
        try:
            with open(greek_yogurt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info("Successfully loaded Greek yogurt market data")
                return data
        except Exception as e:
            logger.error(f"Error loading Greek yogurt data: {e}")
            return {}
    
    def format_market_context(self, product_data: Dict[str, Any]) -> str:
        """
        Format the product data into a readable market context string.
        Only includes market_report content sections.
        
        Args:
            product_data: Raw product data dictionary
            
        Returns:
            Formatted market context string
        """
        if not product_data:
            return "No market information available."
            
        context_parts = []
        
        # Add market report information only
        if "market_report" in product_data:
            market_report = product_data["market_report"]
            
            # Add content sections
            for section_name in ["content", "content_2", "content_3"]:
                if section_name in market_report and market_report[section_name]:
                    context_parts.append(f"\n=== Market Report Section {section_name.upper()} ===")
                    
                    if isinstance(market_report[section_name], list):
                        for item in market_report[section_name]:
                            context_parts.append(f"- {item}")
                    else:
                        context_parts.append(f"- {market_report[section_name]}")
                
        if not context_parts:
            return "No market report information available."
                
        return "\n".join(context_parts)
