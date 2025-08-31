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
                               simulation_stats: Dict[str, Any], 
                               prompt_type: str = "A") -> Dict[str, Any]:
        """
        Generate a summary report of simulation results including demographic breakdowns.
        
        Args:
            results: List of simulation results
            simulation_stats: Statistics from the simulation
            prompt_type: Type of simulation ("A" or "B")
            
        Returns:
            Summary report dictionary with appropriate analysis for the simulation type
        """
        if not results:
            return {"error": "No results to summarize"}
        
        if prompt_type == "B":
            return self._generate_type_b_report(results, simulation_stats)
        elif prompt_type == "C":
            return self._generate_type_c_report(results, simulation_stats)
        else:
            return self._generate_type_a_report(results, simulation_stats)
    
    def _generate_type_a_report(self, results: List[Dict[str, Any]], 
                               simulation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Type A specific report with traditional purchase decision analysis."""
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
        
        # Process Type A results
        for result in successful_results:
            persona = result["persona"]
            decision = result["purchase_decision"]
            
            if decision in ["0", "1"]:
                purchase_counts[decision] += 1
            else:
                purchase_counts["invalid"] += 1
                decision = "invalid"
            
            # Extract demographics for Type A
            gender = persona.get("gender", "Unknown")
            age = persona.get("age", "Unknown")
            
            # Count by gender
            if gender in gender_breakdown:
                gender_breakdown[gender][decision] += 1
            
            # Count by age
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
            "simulation_type": "A",
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
    
    def _generate_type_b_report(self, results: List[Dict[str, Any]], 
                               simulation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Type B specific report with product selection analysis."""
        successful_results = [r for r in results if r["success"]]
        
        # Collect all unique products from all results
        all_products = set()
        for result in successful_results:
            if result.get("product_options_order"):
                all_products.update(result["product_options_order"])
        
        all_products = sorted(list(all_products))
        
        # Initialize product selection counts
        product_counts = {product: 0 for product in all_products}
        product_counts["no_selection"] = 0
        
        # Initialize quantity analysis
        quantity_stats = {}
        
        # Initialize demographic breakdowns by product
        gender_breakdown = {
            "남성": {product: 0 for product in all_products},
            "여성": {product: 0 for product in all_products}
        }
        gender_breakdown["남성"]["no_selection"] = 0
        gender_breakdown["여성"]["no_selection"] = 0
        
        age_breakdown = {
            "20대": {product: 0 for product in all_products},
            "30대": {product: 0 for product in all_products},
            "40대": {product: 0 for product in all_products},
            "50대": {product: 0 for product in all_products},
            "60대": {product: 0 for product in all_products},
            "70대 이상": {product: 0 for product in all_products}
        }
        for age_group in age_breakdown:
            age_breakdown[age_group]["no_selection"] = 0
        
        # Process Type B results
        for result in successful_results:
            persona = result["persona"]
            selected_product = result.get("selected_product")
            selected_quantity = result.get("selected_quantity")
            
            # Count product selections
            if selected_product and selected_product in all_products:
                product_counts[selected_product] += 1
                
                # Analyze quantities
                if selected_quantity:
                    try:
                        qty = int(selected_quantity)
                        if selected_product not in quantity_stats:
                            quantity_stats[selected_product] = []
                        quantity_stats[selected_product].append(qty)
                    except ValueError:
                        pass
            else:
                product_counts["no_selection"] += 1
                selected_product = "no_selection"
            
            # Extract demographics for Type B
            raw_data = persona.get("raw_data", {})
            gender = raw_data.get("성별", "Unknown")
            age = raw_data.get("연령대", "Unknown")
            
            # Map age format for Type B
            age_mapping = {
                "만 19~29세": "20대",
                "만 30~39세": "30대", 
                "만 40~49세": "40대",
                "만 50~59세": "50대",
                "만 60~69세": "60대",
                "만 70세 이상": "70대 이상"
            }
            age = age_mapping.get(age, age)
            
            # Count by demographics
            if gender in gender_breakdown and selected_product in gender_breakdown[gender]:
                gender_breakdown[gender][selected_product] += 1
            
            if age in age_breakdown and selected_product in age_breakdown[age]:
                age_breakdown[age][selected_product] += 1
        
        total_selections = len(successful_results)
        
        # Calculate product selection rates and percentages
        product_analysis = {}
        for product in all_products + ["no_selection"]:
            count = product_counts[product]
            percentage = (count / total_selections * 100) if total_selections > 0 else 0
            
            analysis = {
                "count": count,
                "percentage": round(percentage, 2),
                "selection_rate": round(count / total_selections, 4) if total_selections > 0 else 0
            }
            
            # Add quantity statistics if available
            if product in quantity_stats and quantity_stats[product]:
                quantities = quantity_stats[product]
                analysis["quantity_stats"] = {
                    "average": round(sum(quantities) / len(quantities), 2),
                    "min": min(quantities),
                    "max": max(quantities),
                    "total_quantity": sum(quantities)
                }
            
            product_analysis[product] = analysis
        
        # Calculate demographic breakdown rates
        def calculate_demographic_rates(breakdown_dict):
            rates = {}
            for category, products in breakdown_dict.items():
                total_category = sum(products.values())
                rates[category] = {
                    "total_selections": total_category,
                    "products": {}
                }
                
                for product, count in products.items():
                    rates[category]["products"][product] = {
                        "count": count,
                        "percentage": round((count / total_category * 100) if total_category > 0 else 0, 2),
                        "selection_rate": round(count / total_category, 4) if total_category > 0 else 0
                    }
            return rates
        
        summary = {
            "simulation_type": "B",
            "total_personas": len(results),
            "successful_simulations": len(successful_results),
            "failed_simulations": len(results) - len(successful_results),
            "product_analysis": product_analysis,
            "demographic_breakdown": {
                "by_gender": calculate_demographic_rates(gender_breakdown),
                "by_age": calculate_demographic_rates(age_breakdown)
            },
            "statistics": simulation_stats
        }
        
        return summary
    
    def _generate_type_c_report(self, results: List[Dict[str, Any]], 
                               simulation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Type C specific report with conversion rate analysis by current product."""
        successful_results = [r for r in results if r["success"]]
        conversion_counts = {"0": 0, "1": 0, "invalid": 0}
        
        # Initialize conversion analysis by current product
        current_product_conversion = {}
        
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
        
        # Process Type C results
        for result in successful_results:
            persona = result["persona"]
            decision = result["conversion_decision"]
            current_product = result["current_product"]
            
            # Count overall conversion decisions
            if decision in ["0", "1"]:
                conversion_counts[decision] += 1
            else:
                conversion_counts["invalid"] += 1
                decision = "invalid"
            
            # Track conversion by current product
            if current_product not in current_product_conversion:
                current_product_conversion[current_product] = {"0": 0, "1": 0, "invalid": 0}
            current_product_conversion[current_product][decision] += 1
            
            # Extract demographics for Type C
            raw_data = persona.get("raw_data", {})
            gender = raw_data.get("성별", "Unknown")
            age = raw_data.get("연령대", "Unknown")
            
            # Map age format for Type C
            age_mapping = {
                "만 19~29세": "20대",
                "만 30~39세": "30대", 
                "만 40~49세": "40대",
                "만 50~59세": "50대",
                "만 60~69세": "60대",
                "만 70세 이상": "70대 이상"
            }
            age = age_mapping.get(age, age)
            
            # Count by gender
            if gender in gender_breakdown:
                gender_breakdown[gender][decision] += 1
            
            # Count by age
            if age in age_breakdown:
                age_breakdown[age][decision] += 1
        
        total_valid = conversion_counts["0"] + conversion_counts["1"]
        
        # Calculate conversion rates by current product
        current_product_analysis = {}
        for product, counts in current_product_conversion.items():
            total_product = counts["0"] + counts["1"]
            current_product_analysis[product] = {
                "no_conversion": counts["0"],
                "conversion": counts["1"],
                "invalid_responses": counts["invalid"],
                "total_valid": total_product,
                "conversion_rate": counts["1"] / total_product if total_product > 0 else 0,
                "conversion_percentage": round((counts["1"] / total_product * 100) if total_product > 0 else 0, 2)
            }
        
        # Calculate rates for demographic breakdowns
        def calculate_rates(breakdown_dict):
            rates = {}
            for category, counts in breakdown_dict.items():
                total_category = counts["0"] + counts["1"]
                rates[category] = {
                    "no_conversion": counts["0"],
                    "conversion": counts["1"],
                    "invalid_responses": counts["invalid"],
                    "total_valid": total_category,
                    "conversion_rate": counts["1"] / total_category if total_category > 0 else 0
                }
            return rates
        
        summary = {
            "simulation_type": "C",
            "total_personas": len(results),
            "successful_simulations": len(successful_results),
            "failed_simulations": len(results) - len(successful_results),
            "conversion_decisions": {
                "no_conversion": conversion_counts["0"],
                "conversion": conversion_counts["1"],
                "invalid_responses": conversion_counts["invalid"]
            },
            "overall_conversion_rate": conversion_counts["1"] / total_valid if total_valid > 0 else 0,
            "conversion_by_current_product": current_product_analysis,
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
        simulation_type = summary.get("simulation_type", "A")
        
        logger.info("=== SIMULATION SUMMARY ===")
        logger.info(f"Simulation Type: {simulation_type}")
        logger.info(f"Total Personas: {summary['total_personas']}")
        logger.info(f"Successful Simulations: {summary['successful_simulations']}")
        logger.info(f"Failed Simulations: {summary['failed_simulations']}")
        
        if simulation_type == "A":
            self._print_type_a_summary(summary)
        elif simulation_type == "B":
            self._print_type_b_summary(summary)
        else:  # Type C
            self._print_type_c_summary(summary)
    
    def _print_type_a_summary(self, summary: Dict[str, Any]) -> None:
        """Print Type A specific summary."""
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
    
    def _print_type_b_summary(self, summary: Dict[str, Any]) -> None:
        """Print Type B specific summary with product analysis."""
        # Product selection overview
        logger.info("\n=== PRODUCT SELECTION ANALYSIS ===")
        product_analysis = summary['product_analysis']
        
        for product, data in product_analysis.items():
            logger.info(f"{product}: {data['count']} selections ({data['percentage']:.1f}%)")
            if 'quantity_stats' in data:
                qty_stats = data['quantity_stats']
                logger.info(f"  Quantity - Avg: {qty_stats['average']}, "
                           f"Range: {qty_stats['min']}-{qty_stats['max']}, "
                           f"Total: {qty_stats['total_quantity']}")
        
        # Gender demographic analysis
        logger.info("\n=== DEMOGRAPHIC BREAKDOWN ===")
        logger.info("Gender Analysis:")
        for gender, data in summary['demographic_breakdown']['by_gender'].items():
            logger.info(f"  {gender} (Total: {data['total_selections']}):")
            for product, product_data in data['products'].items():
                if product_data['count'] > 0:
                    logger.info(f"    {product}: {product_data['count']} ({product_data['percentage']:.1f}%)")
        
        # Age demographic analysis
        logger.info("Age Analysis:")
        for age, data in summary['demographic_breakdown']['by_age'].items():
            logger.info(f"  {age} (Total: {data['total_selections']}):")
            for product, product_data in data['products'].items():
                if product_data['count'] > 0:
                    logger.info(f"    {product}: {product_data['count']} ({product_data['percentage']:.1f}%)")
    
    def _print_type_c_summary(self, summary: Dict[str, Any]) -> None:
        """Print Type C specific summary with conversion analysis."""
        logger.info(f"Conversion: {summary['conversion_decisions']['conversion']}")
        logger.info(f"No Conversion: {summary['conversion_decisions']['no_conversion']}")
        logger.info(f"Invalid Responses: {summary['conversion_decisions']['invalid_responses']}")
        logger.info(f"Overall Conversion Rate: {summary['overall_conversion_rate']:.2%}")
        
        # Conversion analysis by current product
        logger.info("\n=== CONVERSION ANALYSIS BY CURRENT PRODUCT ===")
        conversion_analysis = summary['conversion_by_current_product']
        
        for product, data in conversion_analysis.items():
            logger.info(f"{product}:")
            logger.info(f"  Total Valid: {data['total_valid']}")
            logger.info(f"  Conversions: {data['conversion']} ({data['conversion_percentage']:.1f}%)")
            logger.info(f"  No Conversions: {data['no_conversion']}")
            logger.info(f"  Conversion Rate: {data['conversion_rate']:.2%}")
        
        # Display demographic breakdowns
        logger.info("\n=== DEMOGRAPHIC BREAKDOWN ===")
        
        # Gender breakdown
        logger.info("Gender Analysis:")
        for gender, data in summary['demographic_breakdown']['by_gender'].items():
            logger.info(f"  {gender}: Conversion={data['conversion']}, No Conversion={data['no_conversion']}, "
                       f"Rate={data['conversion_rate']:.2%}")
        
        # Age breakdown
        logger.info("Age Analysis:")
        for age, data in summary['demographic_breakdown']['by_age'].items():
            logger.info(f"  {age}: Conversion={data['conversion']}, No Conversion={data['no_conversion']}, "
                       f"Rate={data['conversion_rate']:.2%}")
