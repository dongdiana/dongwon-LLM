#!/usr/bin/env python3
"""
Script to create sub-persona files organized by selected product categories 
from TypeD simulation results.
"""

import json
import os
import argparse
import re
from collections import defaultdict

def load_simulation_results(filepath):
    """Load simulation results from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_original_personas(filepath):
    """Load original persona data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        personas = json.load(f)
    
    # Create a lookup dictionary by UUID
    persona_lookup = {persona['uuid']: persona for persona in personas}
    return persona_lookup

def extract_persona_by_category(simulation_results):
    """Extract persona IDs grouped by their selected product category"""
    category_personas = defaultdict(list)
    
    for result in simulation_results['results']:
        if result.get('success') and result.get('selected_category'):
            persona_id = result['persona_id']
            category = result['selected_category']
            category_personas[category].append(persona_id)
    
    return dict(category_personas)

def sanitize_filename(filename):
    """Sanitize filename by replacing special characters with safe alternatives"""
    # Replace common special characters with safe alternatives
    filename = filename.replace(' ', '_')  # Space to underscore
    filename = filename.replace('/', '_')  # Forward slash
    filename = filename.replace('\\', '_') # Backslash
    filename = filename.replace(':', '_')  # Colon
    filename = filename.replace('*', '_')  # Asterisk
    filename = filename.replace('?', '_')  # Question mark
    filename = filename.replace('"', '_')  # Double quote
    filename = filename.replace('<', '_')  # Less than
    filename = filename.replace('>', '_')  # Greater than
    filename = filename.replace('|', '_')  # Pipe
    
    # Remove any remaining problematic characters using regex
    # Keep only letters, numbers, underscores, hyphens, and Korean characters
    filename = re.sub(r'[^\w\-가-힣]', '_', filename)
    
    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    
    return filename

def create_sub_persona_files(category_personas, persona_lookup, output_dir):
    """Create sub-persona files for each product category"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    created_files = []
    
    for category, persona_ids in category_personas.items():
        # Create filename based on product category (sanitized)
        sanitized_category = sanitize_filename(category)
        filename = f"{sanitized_category}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Collect personas for this category
        personas_for_category = []
        missing_personas = []
        
        for persona_id in persona_ids:
            if persona_id in persona_lookup:
                personas_for_category.append(persona_lookup[persona_id])
            else:
                missing_personas.append(persona_id)
        
        # Write the sub-persona file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(personas_for_category, f, indent=2, ensure_ascii=False)
        
        created_files.append({
            'category': category,
            'filename': filename,
            'persona_count': len(personas_for_category),
            'missing_personas': len(missing_personas)
        })
        
        print(f"Created: {filename}")
        print(f"  - Original category: {category}")
        print(f"  - Personas: {len(personas_for_category)}")
        if missing_personas:
            print(f"  - Missing personas: {len(missing_personas)}")
        print()
    
    return created_files

def main():
    """Main execution function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create sub-persona files organized by selected product categories from TypeD simulation results")
    parser.add_argument("simulation_results_path", help="Path to the simulation results JSON file")
    parser.add_argument("original_personas_path", help="Path to the original personas JSON file")
    parser.add_argument("--output_dir", "-o", default=".", help="Output directory for sub-persona files (default: current directory)")
    
    args = parser.parse_args()
    
    simulation_results_path = args.simulation_results_path
    original_personas_path = args.original_personas_path
    output_dir = args.output_dir
    
    print("Loading simulation results...")
    simulation_results = load_simulation_results(simulation_results_path)
    
    print("Loading original persona data...")
    persona_lookup = load_original_personas(original_personas_path)
    
    print("Extracting personas by selected category...")
    category_personas = extract_persona_by_category(simulation_results)
    
    print(f"Found {len(category_personas)} product categories:")
    for category, persona_ids in category_personas.items():
        print(f"  - {category}: {len(persona_ids)} personas")
    print()
    
    print("Creating sub-persona files...")
    created_files = create_sub_persona_files(category_personas, persona_lookup, output_dir)
    
    print("Summary:")
    print("-" * 50)
    total_personas = sum(info['persona_count'] for info in created_files)
    print(f"Total files created: {len(created_files)}")
    print(f"Total personas processed: {total_personas}")
    print(f"Files saved in: {output_dir}/")
    
    for info in created_files:
        print(f"  - {info['filename']}: {info['persona_count']} personas")
        if info['missing_personas'] > 0:
            print(f"    (Warning: {info['missing_personas']} personas not found in original data)")

if __name__ == "__main__":
    main()
