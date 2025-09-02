"""
Data loader module for persona and product information.
Handles loading JSONL persona files and product JSON data.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonaLoader:
    """Loads persona data from CSV file or detailed JSON persona files."""
    
    def __init__(self, persona_data_dir: str, persona_filename: str = None, sample_size: int = 0):
        self.persona_data_dir = Path(persona_data_dir)
        self.persona_filename = persona_filename
        self.sample_size = sample_size  # 0 means use all personas
        
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
            
            # Apply random sampling if requested
            if self.sample_size > 0 and self.sample_size < len(personas):
                original_count = len(personas)
                random.shuffle(personas)
                personas = personas[:self.sample_size]
                logger.info(f"Random sampling applied: using {len(personas)} out of {original_count} personas")
            
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
    
    def load_detailed_personas(self, product_name: str) -> List[Dict[str, Any]]:
        """
        Load detailed persona data from persona/{product_name}.json file.
        
        Args:
            product_name: Name of the product to load personas for
            
        Returns:
            List of detailed persona dictionaries
        """
        personas = []
        
        # Construct the JSON file path
        persona_json_path = Path("persona") / f"{product_name}.json"
        
        if not persona_json_path.exists():
            logger.warning(f"Detailed persona JSON file does not exist: {persona_json_path}")
            return personas
            
        try:
            # Load JSON file
            with open(persona_json_path, 'r', encoding='utf-8') as f:
                persona_data = json.load(f)
                
            logger.info(f"Loaded detailed persona JSON file with {len(persona_data)} personas from {persona_json_path}")
            
            # Process each persona
            for persona in persona_data:
                # Extract reasoning without "이 페르소나는 " prefix
                reasoning = persona.get("reasoning", "")
                if reasoning.startswith("이 페르소나는 "):
                    reasoning = reasoning[7:]  # Remove "이 페르소나는 " (7 characters)
                
                # Create formatted persona info excluding uuid, segment_key_input, and reasoning
                excluded_fields = {"uuid", "segment_key_input", "reasoning"}
                persona_info_parts = []
                
                for key, value in persona.items():
                    if key not in excluded_fields:
                        persona_info_parts.append(f"{key}: {value}")
                
                persona_info = "\n".join(persona_info_parts)
                
                processed_persona = {
                    "uuid": persona["uuid"],
                    "segment_key_input": persona.get("segment_key_input", ""),
                    "reasoning": reasoning,
                    "persona_info": persona_info,
                    "raw_data": persona  # Keep original data for reference
                }
                
                personas.append(processed_persona)
            
            logger.info(f"Successfully processed {len(personas)} detailed personas")
            
            # Apply random sampling if requested
            if self.sample_size > 0 and self.sample_size < len(personas):
                original_count = len(personas)
                random.shuffle(personas)
                personas = personas[:self.sample_size]
                logger.info(f"Random sampling applied: using {len(personas)} out of {original_count} personas")
            
            return personas
            
        except Exception as e:
            logger.error(f"Error loading detailed persona JSON file: {e}")
            return []


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
    
    def load_type_d_product_data(self, product_filename: str) -> Dict[str, Any]:
        """
        Load TypeD product data from TypeD folder.
        
        Args:
            product_filename: Name of the product data file (with or without .json extension)
        
        Returns:
            Dictionary containing TypeD product data
        """
        # Auto-append .json extension if not present
        if not product_filename.endswith('.json'):
            product_filename += '.json'
            
        type_d_file = self.product_data_dir / "TypeD" / product_filename
        
        if not type_d_file.exists():
            logger.error(f"TypeD product data file not found: {type_d_file}")
            return {}
            
        try:
            with open(type_d_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded TypeD product data from {type_d_file}")
                return data
        except Exception as e:
            logger.error(f"Error loading TypeD product data from {product_filename}: {e}")
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
    
    def get_target_product_info(self, product_data: Dict[str, Any]) -> str:
        """
        Extract target product information for Type C prompts.
        
        Args:
            product_data: Raw product data dictionary
            
        Returns:
            Formatted string with target product name, product_info content, and nutrition_per100
        """
        if not product_data:
            return "No target product information available."
        
        # Find target product (first key that's not "유사제품군" or "market_report")
        target_product_name = None
        for key in product_data.keys():
            if key not in ["유사제품군", "market_report"]:
                target_product_name = key
                break
        
        if not target_product_name or target_product_name not in product_data:
            return "Target product not found."
        
        target_product = product_data[target_product_name]
        
        # Extract content from product_info
        product_info_content = []
        if "product_info" in target_product and "content" in target_product["product_info"]:
            content = target_product["product_info"]["content"]
            if isinstance(content, list):
                product_info_content = content
            else:
                product_info_content = [content]
        
        # Extract nutrition info
        nutrition_info = target_product.get("nutrition_per100", {})
        
        # Format target product information
        product_info_str = " ".join(product_info_content) if product_info_content else "No product info available"
        
        # Handle nested nutrition structure if exists
        if nutrition_info:
            # Check if nutrition_per100 has nested structure (like "리챔 오믈레햄 200g")
            if any(isinstance(v, dict) for v in nutrition_info.values()):
                # Take the first nested nutrition data
                for nutrition_key, nutrition_data in nutrition_info.items():
                    if isinstance(nutrition_data, dict):
                        nutrition_str = ", ".join([f"{k}: {v}" for k, v in nutrition_data.items()])
                        break
                else:
                    nutrition_str = "No nutrition info available"
            else:
                # Direct nutrition data
                nutrition_str = ", ".join([f"{k}: {v}" for k, v in nutrition_info.items()])
        else:
            nutrition_str = "No nutrition info available"
        
        return f"{target_product_name}: {product_info_str}, {nutrition_str}"
    
    def get_current_product_info(self, product_data: Dict[str, Any], current_product_name: str) -> str:
        """
        Extract current product information for Type C prompts based on persona's existing product.
        
        Args:
            product_data: Raw product data dictionary
            current_product_name: Name of the current product used by persona
            
        Returns:
            Formatted string with current product name, product_info, and nutrition_per100
        """
        if not product_data or not current_product_name:
            return "No current product information available."
        
        # Look for the current product in similar products section
        if "유사제품군" not in product_data:
            return "No similar products section found."
        
        similar_products = product_data["유사제품군"]
        
        # Find matching product in similar products
        current_product_data = None
        for product_name, product_info in similar_products.items():
            if product_name == current_product_name:
                current_product_data = product_info
                break
        
        if not current_product_data:
            return f"Current product '{current_product_name}' not found in similar products."
        
        # Extract product_info (list format for similar products)
        product_info_content = current_product_data.get("product_info", [])
        if isinstance(product_info_content, list):
            product_info_str = " ".join(product_info_content)
        else:
            product_info_str = str(product_info_content)
        
        # Extract nutrition info
        nutrition_info = current_product_data.get("nutrition_per100", {})
        nutrition_str = ", ".join([f"{k}: {v}" for k, v in nutrition_info.items()]) if nutrition_info else "No nutrition info available"
        
        return f"{current_product_name}: {product_info_str}, {nutrition_str}"
    
    def generate_product_options(self, product_data: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Generate randomized product options from product data for Type B prompts.
        
        Args:
            product_data: Raw product data dictionary
            
        Returns:
            Tuple of (formatted_options_string, list_of_product_names_in_order)
        """
        if not product_data:
            return "No product options available.", []
        
        options = []
        product_names = []
        
        # Find target product (first key that's not "유사제품군" or "market_report")
        target_product_name = None
        for key in product_data.keys():
            if key not in ["유사제품군", "market_report"]:
                target_product_name = key
                break
        
        # Add target product
        if target_product_name and target_product_name in product_data:
            target_product = product_data[target_product_name]
            
            # Extract content from product_info
            product_info_content = []
            if "product_info" in target_product and "content" in target_product["product_info"]:
                content = target_product["product_info"]["content"]
                if isinstance(content, list):
                    product_info_content = content
                else:
                    product_info_content = [content]
            
            # Extract nutrition info
            nutrition_info = target_product.get("nutrition_per100", {})
            
            # Format target product option
            product_info_str = " ".join(product_info_content) if product_info_content else "No product info available"
            nutrition_str = ", ".join([f"{k}: {v}" for k, v in nutrition_info.items()]) if nutrition_info else "No nutrition info available"
            
            options.append({
                "name": target_product_name,
                "info": f"product_info: {product_info_str}, nutrition_per100: {nutrition_str}"
            })
            product_names.append(target_product_name)
        
        # Add similar products
        if "유사제품군" in product_data:
            similar_products = product_data["유사제품군"]
            
            for similar_product_name, similar_product_data in similar_products.items():
                # Extract product_info (list format for similar products)
                product_info_content = similar_product_data.get("product_info", [])
                if isinstance(product_info_content, list):
                    product_info_str = " ".join(product_info_content)
                else:
                    product_info_str = str(product_info_content)
                
                # Extract nutrition info
                nutrition_info = similar_product_data.get("nutrition_per100", {})
                nutrition_str = ", ".join([f"{k}: {v}" for k, v in nutrition_info.items()]) if nutrition_info else "No nutrition info available"
                
                options.append({
                    "name": similar_product_name,
                    "info": f"product_info: {product_info_str}, nutrition_per100: {nutrition_str}"
                })
                product_names.append(similar_product_name)
        
        # Randomize the order
        combined = list(zip(options, product_names))
        random.shuffle(combined)
        options, product_names = zip(*combined) if combined else ([], [])
        
        # Format options string
        formatted_options = []
        for i, option in enumerate(options, 1):
            formatted_options.append(f"{i}. ({option['name']}) {option['info']}")
        
        options_string = "\n".join(formatted_options)
        
        return options_string, list(product_names)
    
    def get_type_d_product_name(self, type_d_data: Dict[str, Any]) -> str:
        """
        Extract the top-level product name from TypeD data.
        
        Args:
            type_d_data: TypeD product data dictionary
            
        Returns:
            Top-level product name (first key in the JSON)
        """
        if not type_d_data:
            return "Unknown Product"
        
        # Get the first key (top-level schema)
        for key in type_d_data.keys():
            return key
        
        return "Unknown Product"
    
    def get_type_d_product_info(self, type_d_data: Dict[str, Any]) -> str:
        """
        Extract product_info content from TypeD data.
        
        Args:
            type_d_data: TypeD product data dictionary
            
        Returns:
            Formatted product info content (excluding 출처)
        """
        if not type_d_data:
            return "No product information available."
        
        # Get the first product (top-level)
        product_name = self.get_type_d_product_name(type_d_data)
        if product_name not in type_d_data:
            return "No product information available."
        
        product_data = type_d_data[product_name]
        
        # Extract product_info content
        if "product_info" in product_data and "content" in product_data["product_info"]:
            content = product_data["product_info"]["content"]
            if isinstance(content, list):
                return "\n".join([f"- {item}" for item in content])
            else:
                return f"- {content}"
        
        return "No product information available."
    
    def get_type_d_product_options(self, type_d_data: Dict[str, Any]) -> str:
        """
        Extract category information as product options for TypeD.
        
        Args:
            type_d_data: TypeD product data dictionary
            
        Returns:
            Formatted product options string
        """
        if not type_d_data:
            return "No product options available."
        
        # Get the first product (top-level)
        product_name = self.get_type_d_product_name(type_d_data)
        if product_name not in type_d_data:
            return "No product options available."
        
        product_data = type_d_data[product_name]
        
        # Extract category as options
        if "category" in product_data:
            categories = product_data["category"]
            if isinstance(categories, list):
                formatted_options = []
                for i, category in enumerate(categories, 1):
                    formatted_options.append(f"{i}. {category}")
                return "\n".join(formatted_options)
            else:
                return f"1. {categories}"
        
        return "No product options available."
