"""
Report generation module for LLM simulation results.
Handles saving results and generating summary reports with demographic breakdowns.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class SimulationReporter:
    """
    Handles generation and saving of simulation reports and summaries.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the reporter with results directory.
        
        Args:
            results_dir: Directory to save results files
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def save_results(self, results: List[Dict[str, Any]], simulation_stats: Dict[str, Any], 
                     model_name: str, product_name: str, filename: Optional[str] = None) -> str:
        """
        Save simulation results to JSON file.
        
        Args:
            results: List of simulation results
            simulation_stats: Statistics from the simulation
            model_name: Name of the LLM model used
            product_name: Name of the product being analyzed
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.json"
        
        file_path = self.results_dir / filename
        
        # Prepare output data
        output_data = {
            "metadata": {
                "simulation_time": datetime.now().isoformat(),
                "model_used": model_name,
                "product_name": product_name,
                "total_personas": len(results),
                "statistics": simulation_stats
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
    
    def save_summary_report(self, summary: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save summary report to JSON file.
        
        Args:
            summary: Summary report dictionary
            filename: Optional custom filename
            
        Returns:
            Path to saved summary file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_report_{timestamp}.json"
        
        file_path = self.results_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Summary report saved to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving summary report: {e}")
            raise
    
    def generate_summary_report(self, results: List[Dict[str, Any]], 
                               simulation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary report of simulation results including demographic breakdowns.
        
        Args:
            results: List of simulation results
            simulation_stats: Statistics from the simulation
            
        Returns:
            Summary report dictionary with gender/age breakdowns
        """
        if not results:
            return {"error": "No results to summarize"}
        
        successful_results = [r for r in results if r["success"]]
        purchase_counts = {"0": 0, "1": 0, "invalid": 0}
        
        # Initialize demographic breakdowns
        gender_breakdown = {
            "남성": {"0": 0, "1": 0, "invalid": 0},
            "여성": {"0": 0, "1": 0, "invalid": 0}
        }
        
        age_breakdown = {
            "20대": {"0": 0, "1": 0, "invalid": 0},
            "30대": {"0": 0, "1": 0, "invalid": 0},
            "40대": {"0": 0, "1": 0, "invalid": 0},
            "50대": {"0": 0, "1": 0, "invalid": 0},
            "60대": {"0": 0, "1": 0, "invalid": 0},
            "70대 이상": {"0": 0, "1": 0, "invalid": 0}
        }
        
        # Process results and count decisions by demographics
        for result in successful_results:
            decision = result["purchase_decision"]
            persona = result["persona"]
            
            # Count overall decisions
            if decision in ["0", "1"]:
                purchase_counts[decision] += 1
            else:
                purchase_counts["invalid"] += 1
                decision = "invalid"
            
            # Count by gender
            gender = persona.get("gender", "Unknown")
            if gender in gender_breakdown:
                gender_breakdown[gender][decision] += 1
            
            # Count by age
            age = persona.get("age", "Unknown")
            if age in age_breakdown:
                age_breakdown[age][decision] += 1
        
        total_valid = purchase_counts["0"] + purchase_counts["1"]
        
        # Calculate rates for demographic breakdowns
        def calculate_rates(breakdown_dict):
            rates = {}
            for category, counts in breakdown_dict.items():
                total_category = counts["0"] + counts["1"]
                rates[category] = {
                    "will_not_purchase": counts["0"],
                    "will_purchase": counts["1"],
                    "invalid_responses": counts["invalid"],
                    "total_valid": total_category,
                    "purchase_rate": counts["1"] / total_category if total_category > 0 else 0
                }
            return rates
        
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
            "demographic_breakdown": {
                "by_gender": calculate_rates(gender_breakdown),
                "by_age": calculate_rates(age_breakdown)
            },
            "statistics": simulation_stats
        }
        
        return summary
    
    def print_summary_log(self, summary: Dict[str, Any]) -> None:
        """
        Print summary report to logger for console output.
        
        Args:
            summary: Summary report dictionary
        """
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
