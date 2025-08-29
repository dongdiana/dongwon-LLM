"""
Main entry point for the Product Purchase Decision Simulation.
Orchestrates the loading of persona data, product information, and runs LLM simulations.
"""

import asyncio
import logging
import sys
import json
import yaml
from pathlib import Path

from src.loader import PersonaLoader
from src.simulator import PersonaSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('simulation.log')
    ]
)

logger = logging.getLogger(__name__)





async def main():
    """
    Main execution function.
    """
    try:
        logger.info("Starting Product Purchase Decision Simulation")
        
        # Initialize configuration
        config_path = "config.yaml"
        prompt_path = "prompt.yaml"
        
        # Check if config files exist
        if not Path(config_path).exists():
            logger.error(f"Configuration file not found: {config_path}")
            return
        
        if not Path(prompt_path).exists():
            logger.error(f"Prompt file not found: {prompt_path}")
            return
        
        # Load configuration
        logger.info("Loading configuration...")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Initialize loaders
        logger.info("Initializing data loaders...")
        persona_data_dir = config["paths"]["persona_data_dir"]
        persona_filename = config["persona"]["filename"]
        persona_loader = PersonaLoader(persona_data_dir, persona_filename)
        
        # Load personas
        logger.info("Loading persona data...")
        personas = persona_loader.load_all_personas()
        
        if not personas:
            logger.error("No valid persona data found. Cannot proceed with simulation.")
            return
        
        logger.info(f"Loaded {len(personas)} personas")
        
        # Initialize simulator
        logger.info("Initializing LLM simulator...")
        simulator = PersonaSimulator(config_path, prompt_path)
        
        # Run simulation
        logger.info("Starting persona simulations...")
        results = await simulator.simulate_all_personas(personas)
        
        # Save results
        logger.info("Saving simulation results...")
        results_file = simulator.save_results(results)
        
        # Generate and display summary
        summary = simulator.generate_summary_report(results)
        
        logger.info("=== SIMULATION SUMMARY ===")
        logger.info(f"Total Personas: {summary['total_personas']}")
        logger.info(f"Successful Simulations: {summary['successful_simulations']}")
        logger.info(f"Failed Simulations: {summary['failed_simulations']}")
        logger.info(f"Will Purchase: {summary['purchase_decisions']['will_purchase']}")
        logger.info(f"Will Not Purchase: {summary['purchase_decisions']['will_not_purchase']}")
        logger.info(f"Invalid Responses: {summary['purchase_decisions']['invalid_responses']}")
        logger.info(f"Purchase Rate: {summary['purchase_rate']:.2%}")
        
        # Display demographic breakdowns
        logger.info("\n=== DEMOGRAPHIC BREAKDOWN ===")
        
        # Gender breakdown
        logger.info("Gender Analysis:")
        for gender, data in summary['demographic_breakdown']['by_gender'].items():
            logger.info(f"  {gender}: Purchase={data['will_purchase']}, No Purchase={data['will_not_purchase']}, "
                       f"Rate={data['purchase_rate']:.2%}")
        
        # Age breakdown
        logger.info("Age Analysis:")
        for age, data in summary['demographic_breakdown']['by_age'].items():
            logger.info(f"  {age}: Purchase={data['will_purchase']}, No Purchase={data['will_not_purchase']}, "
                       f"Rate={data['purchase_rate']:.2%}")
        
        logger.info(f"Results saved to: {results_file}")
        
        # Save summary report
        summary_file = Path("results") / f"summary_report_{results_file.split('_')[-1]}"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary report saved to: {summary_file}")
        logger.info("Simulation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
