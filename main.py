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
from src.report import SimulationReporter

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
        
        # Initialize simulator and reporter
        logger.info("Initializing LLM simulator...")
        simulator = PersonaSimulator(config_path, prompt_path)
        reporter = SimulationReporter(config["paths"]["results_dir"])
        
        # Run simulation
        logger.info("Starting persona simulations...")
        results = await simulator.simulate_all_personas(personas)
        
        # Save results
        logger.info("Saving simulation results...")
        results_file = reporter.save_results(
            results, 
            simulator.simulation_stats, 
            simulator.config["llm"]["model_name"], 
            simulator.get_product_name()
        )
        
        # Generate and display summary
        summary = reporter.generate_summary_report(results, simulator.simulation_stats)
        
        # Print summary to log
        reporter.print_summary_log(summary)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Save summary report
        timestamp = results_file.split('_')[-1].replace('.json', '')
        summary_file = reporter.save_summary_report(summary, f"summary_report_{timestamp}.json")
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
