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
    
    def __init__(self, persona_data_dir: str, persona_filename: str = None):
        self.persona_data_dir = Path(persona_data_dir)
        self.persona_filename = persona_filename
        
    def load_all_personas(self) -> List[Dict[str, Any]]:
        """
        Load all personas from CSV file.
        
        Returns:
            List of persona dictionaries
        """
        personas = []
        
        # Construct the CSV file path
        if self.persona_filename:
            # Auto-append .csv extension if not present
            filename = self.persona_filename
            if not filename.endswith('.csv'):
                filename += '.csv'
            persona_csv_path = self.persona_data_dir / filename
        else:
            # Fallback to default filename
            persona_csv_path = self.persona_data_dir / "persona.csv"
        
        if not persona_csv_path.exists():
            logger.error(f"Persona CSV file does not exist: {persona_csv_path}")
            return personas
            
        try:
            # Load CSV file
            df = pd.read_csv(persona_csv_path, encoding='utf-8')
            logger.info(f"Loaded CSV file with {len(df)} rows from {persona_csv_path}")
            
            # Convert to list of dictionaries
            for _, row in df.iterrows():
                persona = {
                    "id": str(row["id"]),
                    "gender": row["성별"],
                    "age": str(row["연령대"]),
                    "household_size": str(row["가구원수"]),
                    "income": row["소득구간"]
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
        required_fields = ["id", "gender", "age", "household_size", "income"]
        
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
        
    def load_product_data(self, product_filename: str) -> Dict[str, Any]:
        """
        Load product market information for any product.
        
        Args:
            product_filename: Name of the product data file (with or without .json extension)
        
        Returns:
            Dictionary containing product market data
        """
        # Auto-append .json extension if not present
        if not product_filename.endswith('.json'):
            product_filename += '.json'
            
        product_file = self.product_data_dir / product_filename
        
        if not product_file.exists():
            logger.error(f"Product data file not found: {product_file}")
            return {}
            
        try:
            with open(product_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded product data from {product_filename}")
                return data
        except Exception as e:
            logger.error(f"Error loading product data from {product_filename}: {e}")
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
        
        # Add market report information only (no search data)
        if "market_report" in product_data:
            market_report = product_data["market_report"]
            
            # Add content sections
            for section_name in ["content"]:
                if section_name in market_report and market_report[section_name]:                    
                    if isinstance(market_report[section_name], list):
                        for item in market_report[section_name]:
                            context_parts.append(f"- {item}")
                    else:
                        context_parts.append(f"- {market_report[section_name]}")
                
        if not context_parts:
            return "No market report information available."
                
        return "\n".join(context_parts)
    
    def format_search_summary(self, naver_trend_data: Dict[str, Any], product_name: str) -> str:
        """
        Format Naver trend search data into a readable search summary string.
        Separate from market context as per requirements.
        
        Args:
            naver_trend_data: Naver trend data dictionary
            product_name: Name of the product for context
            
        Returns:
            Formatted search summary string
        """
        if not naver_trend_data:
            return "No search trend data available."
            
        try:
            from collections import defaultdict
            
            def collapse(series_by_time):
                """Collapse time series data into totals and shares."""
                totals = defaultdict(float)
                
                # Sum all values across all time periods
                for time_period, segments in series_by_time.items():
                    for segment, value in segments.items():
                        totals[segment] += float(value)
                
                # Calculate percentages
                total_sum = sum(totals.values())
                shares = {k: (totals[k] / total_sum * 100 if total_sum else 0.0) for k in totals}
                return shares
            
            # Process gender and age data
            gender_shares = collapse(naver_trend_data.get("gender", {}))
            age_shares = collapse(naver_trend_data.get("age", {}))
            
            # Format percentage
            def pct(x): 
                return f"{x:.1f}%"
            
            # Build search summary text
            summary_parts = []
            summary_parts.append(f"{product_name}의 성별에 따른 검색량 비율은 남성 {pct(gender_shares.get('m', 0))}, 여성 {pct(gender_shares.get('f', 0))} 입니다.")
            summary_parts.append(f"{product_name}의 연령에 따른 검색량 비율은 20대 {pct(age_shares.get('20', 0))}, 30대 {pct(age_shares.get('30', 0))}, "
                                f"40대 {pct(age_shares.get('40', 0))}, 50대 {pct(age_shares.get('50', 0))}, 60대 이상 {pct(age_shares.get('60', 0))} 입니다.")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.warning(f"Error formatting search summary: {e}")
            return "Error processing search trend data."
    
    def load_naver_trend_data(self, product_filename: str) -> Dict[str, Any]:
        """
        Load Naver trend data for the specified product.
        
        Args:
            product_filename: Name of the product file (with or without .json extension)
            
        Returns:
            Dictionary containing Naver trend data
        """
        # Auto-append .json extension if not present
        if not product_filename.endswith('.json'):
            product_filename += '.json'
            
        trend_path = Path("data/naver_trend") / product_filename
        
        if not trend_path.exists():
            logger.debug(f"No Naver trend data found at {trend_path}")
            return {}
            
        try:
            with open(trend_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded Naver trend data from {product_filename}")
                return data
        except Exception as e:
            logger.warning(f"Error loading Naver trend data from {product_filename}: {e}")
            return {}
