#!/usr/bin/env python3
"""
Preprocessing script to convert raw Naver trend data to processed format.

Converts data/naver_trend/raw/{product}.json to data/naver_trend/{product}.json

The raw format has:
- gender.female and gender.male sections
- gender.gender section with f/m keys (this is what we want)
- age section with numeric age group keys

The processed format needs:
- gender section with f/m keys (from gender.gender)
- age section (unchanged)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_naver_trend_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw Naver trend data to processed format.
    
    Args:
        raw_data: Raw trend data from Naver
        
    Returns:
        Processed trend data in the format expected by simulator
    """
    processed_data = {}
    
    # Extract gender data from the nested gender.gender section
    if "gender" in raw_data and "gender" in raw_data["gender"]:
        processed_data["gender"] = raw_data["gender"]["gender"]
        logger.info("Extracted gender data from nested gender.gender section")
    else:
        logger.warning("No nested gender.gender section found in raw data")
        processed_data["gender"] = {}
    
    # Copy age data directly (should be the same format)
    if "age" in raw_data:
        processed_data["age"] = raw_data["age"]
        logger.info("Copied age data directly from raw data")
    else:
        logger.warning("No age section found in raw data")
        processed_data["age"] = {}
    
    return processed_data


def process_single_file(product_name: str) -> bool:
    """
    Process a single product's trend data.
    
    Args:
        product_name: Name of the product (without .json extension)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Define file paths
        raw_path = Path("data/naver_trend/raw") / f"{product_name}.json"
        output_path = Path("data/naver_trend") / f"{product_name}.json"
        
        # Check if raw file exists
        if not raw_path.exists():
            logger.error(f"Raw file not found: {raw_path}")
            return False
        
        # Read raw data
        logger.info(f"Reading raw data from {raw_path}")
        with open(raw_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Process the data
        processed_data = preprocess_naver_trend_data(raw_data)
        
        # Ensure output directory exists
        output_path.parent.mkdir(exist_ok=True)
        
        # Write processed data
        logger.info(f"Writing processed data to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Successfully processed {product_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {product_name}: {e}")
        return False


def process_all_files() -> None:
    """Process all raw files in the data/naver_trend/raw/ directory."""
    raw_dir = Path("data/naver_trend/raw")
    
    if not raw_dir.exists():
        logger.error(f"Raw directory not found: {raw_dir}")
        return
    
    # Find all JSON files in raw directory
    raw_files = list(raw_dir.glob("*.json"))
    
    if not raw_files:
        logger.warning(f"No JSON files found in {raw_dir}")
        return
    
    logger.info(f"Found {len(raw_files)} files to process")
    
    successful = 0
    failed = 0
    
    for raw_file in raw_files:
        product_name = raw_file.stem  # filename without extension
        logger.info(f"Processing {product_name}...")
        
        if process_single_file(product_name):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Processing complete. Successful: {successful}, Failed: {failed}")


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1:
        # Process specific product
        product_name = sys.argv[1]
        logger.info(f"Processing single product: {product_name}")
        
        if process_single_file(product_name):
            logger.info("Processing completed successfully")
        else:
            logger.error("Processing failed")
            sys.exit(1)
    else:
        # Process all files
        logger.info("Processing all files in data/naver_trend/raw/")
        process_all_files()


if __name__ == "__main__":
    main()