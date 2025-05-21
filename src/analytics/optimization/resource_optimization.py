#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resource optimization module for AI for Sustainable Ecosystems project.
This script implements optimization models for sustainable resource management.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from dotenv import load_dotenv

# Optimization libraries
import pulp
import cvxpy as cp
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resource_optimization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("resource_optimization")

# Load environment variables
load_dotenv()

# Set paths
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results"))


def load_waste_data(file_path=None):
    """
    Load waste management data from file.
    
    Args:
        file_path (str, optional): Path to waste data file. If None, use default.
    
    Returns:
        pd.DataFrame: Waste management data
    """
    if file_path is None:
        file_path = DATA_DIR / "waste" / "waste_statistics.csv"
    
    logger.info(f"Loading waste data from: {file_path}")
    
    try:
        # Try loading the data
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded waste data with {len(df)} records")
        return df
    
    except FileNotFoundError:
        logger.warning(f"Waste data file not found: {file_path}")
        logger.info("Creating sample waste data for demonstration")
        return create_sample_waste_data()
    
    except Exception as e:
        logger.error(f"Error loading waste data: {str(e)}")
        logger.info("Creating sample waste data for demonstration")
        return create_sample_waste_data()


def create_sample_waste_data():
    """
    Create sample waste management data for demonstration purposes.
    
    Returns:
        pd.DataFrame: Sample waste management data
    """
    # Sample regions
    regions = ['Region_A', 'Region_B', 'Region_C', 'Region_D', 'Region_E']
    n_regions = len(regions)
    
    # Sample waste types
    waste_types = ['Organic', 'Plastic', 'Paper', 'Metal', 'Glass', 'Other']
    n_waste_types = len(waste_types)
    
    # Generate sample data
    np.random.seed(42)  # For reproducibility
    
    # Waste generation rates (tons per day)
    waste_generation = np.random.uniform(10, 100, n_regions * n_waste_types).reshape(n_regions, n_waste_types)
    
    # Recycling facility capacities (tons per day)
    recycling_capacity = np.random.uniform(5, 50, n_regions * n_waste_types).reshape(n_regions, n_waste_types)
    
    # Transportation costs ($ per ton per km)
    transport_cost = np.random.uniform(0.1, 0.5, n_regions * n_regions).reshape(n_regions, n_regions)
    np.fill_diagonal(transport_cost, 0)  # No cost for same region
    
    # Processing costs ($ per ton)
    processing_cost = np.random.uniform(20, 100, n_waste_types)
    
    # Environmental impact reduction (CO2 equivalent tons saved per ton recycled)
    env_impact = np.random.uniform(0.5, 2.0, n_waste_types)
    
    # Create DataFrame for waste generation
    waste_df = pd.DataFrame(index=regions)
    for i, waste_type in enumerate(waste_types):
        waste_df[f"{waste_type}_Generation"] = waste_generation[:, i]
        waste_df[f"{waste_type}_RecyclingCapacity"] = recycling_capacity[:, i]
    
    # Add transport cost matrix
    transport_df = pd.DataFrame(transport_cost, index=regions, columns=regions)
    
    # Add processing costs and environmental impact
    costs_df = pd.DataFrame({
        'WasteType': waste_types,
        'ProcessingCost': processing_cost,
        'EnvironmentalImpact': env_impact
    })
    
    # Current recycling rates (%)
    waste_df['CurrentRecyclingRate'] = np.random.uniform(0.1, 0.6, n_regions)
    
    # Distance between regions (km)
    distance_df = pd.DataFrame(np.random.uniform(10, 200, n_regions * n_regions).reshape(n_regions, n_regions), 
                              index=regions, columns=regions)
    np.fill_diagonal(distance_df.values, 0)
    
    logger.info(f"Created sample waste data for {n_regions} regions and {n_waste_types} waste types")
    
    return {
        'waste_generation': waste_df,
        'transport_cost': transport_df,
        'processing_costs': costs_df,
        'distances': distance_df
    }


def optimize_waste_allocation(data):
    """
    Optimize waste allocation to maximize recycling while minimizing costs.
    
    Args:
        data (dict): Dictionary containing waste data
        
    Returns:
        dict: Optimization results
    """
    logger.info("Starting waste allocation optimization")
    
    # Extract data
    waste_df = data['waste_generation']
    transport_df = data['transport_cost']
    costs_df = data['processing_costs']
    distances = data['distances']
    
    regions = waste_df.index.tolist()
    n_regions = len(regions)
    
    # Extract waste types
    waste_types = [col.split('_')[0] for col in waste_df.columns if col.endswith('Generation')]
    n_waste_types = len(waste_types)
    
    # Create PuLP optimization model
    model = pulp.LpProblem("Waste_Allocation_Optimization", pulp.LpMaximize)
    
    # Decision variables: amount of waste type w from region i to region j
    waste_vars = {}
    for i in range(n_regions):
        for j in range(n_regions):
            for w in range(n_waste_types):
                waste_vars[(i, j, w)] = pulp.LpVariable(
                    f"Waste_{i}_{j}_{w}", 
                    lowBound=0,
                    cat='Continuous'
                )
    
    # Extract data into NumPy arrays for easier calculation
    waste_generation = np.zeros((n_regions, n_waste_types))
    recycling_capacity = np.zeros((n_regions, n_waste_types))
    
    for w, waste_type in enumerate(waste_types):
        waste_generation[:, w] = waste_df[f"{waste_type}_Generation"].values
        recycling_capacity[:, w] = waste_df[f"{waste_type}_RecyclingCapacity"].values
    
    # Environmental impact (CO2e saved per ton recycled)
    env_impact = costs_df['EnvironmentalImpact'].values
    
    # Processing costs ($ per ton)
    proc_costs = costs_df['ProcessingCost'].values
    
    # Objective function: Maximize environmental benefit minus costs
    env_benefit_expr = 0
    cost_expr = 0
    
    for i in range(n_regions):
        for j in range(n_regions):
            for w in range(n_waste_types):
                # Environmental benefit
                env_benefit_expr += waste_vars[(i, j, w)] * env_impact[w]
                
                # Transportation cost
                transport_cost = distances.iloc[i, j] * transport_df.iloc[i, j]
                cost_expr += waste_vars[(i, j, w)] * transport_cost
                
                # Processing cost
                cost_expr += waste_vars[(i, j, w)] * proc_costs[w]
    
    # Combine into objective (we want to maximize benefit - cost)
    model += env_benefit_expr - 0.01 * cost_expr
    
    # Constraint 1: Cannot exceed waste generation in source region
    for i in range(n_regions):
        for w in range(n_waste_types):
            model += (
                pulp.lpSum([waste_vars[(i, j, w)] for j in range(n_regions)]) <= waste_generation[i, w],
                f"Waste_Generation_Constraint_{i}_{w}"
            )
    
    # Constraint 2: Cannot exceed recycling capacity in destination region
    for j in range(n_regions):
        for w in range(n_waste_types):
            model += (
                pulp.lpSum([waste_vars[(i, j, w)] for i in range(n_regions)]) <= recycling_capacity[j, w],
                f"Recycling_Capacity_Constraint_{j}_{w}"
            )
    
    # Solve the model
    logger.info("Solving optimization model...")
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    logger.info(f"Optimization status: {pulp.LpStatus[model.status]}")
    
    # Extract results if optimal
    if model.status == pulp.LpStatusOptimal:
        # Create results dictionary
        results = {
            'status': 'Optimal',
            'objective_value': pulp.value(model.objective),
            'allocation': np.zeros((n_regions, n_regions, n_waste_types)),
            'total_recycled': np.zeros((n_waste_types)),
            'recycling_rate': np.zeros((n_regions)),
            'environmental_benefit': 0,
            'total_cost': 0
        }
        
        # Extract allocation
        for i in range(n_regions):
            for j in range(n_regions):
                for w in range(n_waste_types):
                    value = waste_vars[(i, j, w)].value()
                    if value > 1e-5:  # Ignore very small values (numerical precision)
                        results['allocation'][i, j, w] = value
                        results['total_recycled'][w] += value
        
        # Calculate recycling rate per region
        for i in range(n_regions):
            total_waste_generated = sum(waste_generation[i, :])
            total_waste_recycled = sum(sum(results['allocation'][i, j, w] for j in range(n_regions)) for w in range(n_waste_types))
            if total_waste_generated > 0:
                results['recycling_rate'][i] = total_waste_recycled / total_waste_generated
        
        # Calculate environmental benefit
        for w in range(n_waste_types):
            results['environmental_benefit'] += results['total_recycled'][w] * env_impact[w]
        
        # Calculate total cost
        for i in range(n_regions):
            for j in range(n_regions):
                for w in range(n_waste_types):
                    transport_cost = distances.iloc[i, j] * transport_df.iloc[i, j]
                    value = results['allocation'][i, j, w]
                    results['total_cost'] += value * (transport_cost + proc_costs[w])
        
        return results
    else:
        logger.error("Optimization did not find an optimal solution")
        return {'status': 'Failed'}


def visualize_optimization_results(results, data, save_path=None):
    """
    Visualize optimization results.
    
    Args:
        results (dict): Optimization results
        data (dict): Original data
        save_path (str, optional): Path to save visualizations
    """
    if results['status'] != 'Optimal':
        logger.error("Cannot visualize: optimization did not find an optimal solution")
        return
    
    waste_df = data['waste_generation']
    regions = waste_df.index.tolist()
    waste_types = [col.split('_')[0] for col in waste_df.columns if col.endswith('Generation')]
    
    # Set up directory for saving figures
    if save_path is None:
        save_path = RESULTS_DIR / "figures"
        os.makedirs(save_path, exist_ok=True)
    
    # 1. Plot recycling rates before and after optimization
    plt.figure(figsize=(10, 6))
    x = np.arange(len(regions))
    width = 0.35
    
    current_rates = waste_df['CurrentRecyclingRate'].values
    optimized_rates = results['recycling_rate']
    
    plt.bar(x - width/2, current_rates, width, label='Current')
    plt.bar(x + width/2, optimized_rates, width, label='Optimized')
    
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Recycling Rate', fontsize=12)
    plt.title('Recycling Rates: Current vs. Optimized', fontsize=14)
    plt.xticks(x, regions)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, "recycling_rates_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Plot waste allocation flows
    allocation = results['allocation']
    n_regions = len(regions)
    n_waste_types = len(waste_types)
    
    # Create a heatmap of total waste flow between regions
    total_flow = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            total_flow[i, j] = sum(allocation[i, j, w] for w in range(n_waste_types))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(total_flow, annot=True, fmt=".1f", cmap="YlGnBu",
                xticklabels=regions, yticklabels=regions)
    plt.title('Waste Flow Between Regions (tons/day)', fontsize=14)
    plt.xlabel('Destination Region', fontsize=12)
    plt.ylabel('Source Region', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, "waste_flow_heatmap.png"), dpi=300)
    plt.close()
    
    # 3. Plot recycled amount by waste type
    plt.figure(figsize=(10, 6))
    total_recycled = results['total_recycled']
    
    # Get total waste generated by type for comparison
    total_generated = np.zeros(n_waste_types)
    for w, waste_type in enumerate(waste_types):
        total_generated[w] = waste_df[f"{waste_type}_Generation"].sum()
    
    x = np.arange(len(waste_types))
    width = 0.35
    
    plt.bar(x - width/2, total_generated, width, label='Generated')
    plt.bar(x + width/2, total_recycled, width, label='Recycled')
    
    plt.xlabel('Waste Type', fontsize=12)
    plt.ylabel('Amount (tons/day)', fontsize=12)
    plt.title('Waste Generated vs. Recycled by Type', fontsize=14)
    plt.xticks(x, waste_types)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, "waste_by_type.png"), dpi=300)
    plt.close()
    
    # 4. Plot a summary of environmental benefit and cost
    plt.figure(figsize=(8, 5))
    
    metrics = ['Environmental Benefit\n(CO2e tons saved/day)', 'Total Cost\n($/day)']
    values = [results['environmental_benefit'], results['total_cost']]
    
    plt.bar(metrics, values, color=['green', 'orange'])
    plt.ylabel('Value', fontsize=12)
    plt.title('Optimization Key Metrics', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "optimization_metrics.png"), dpi=300)
    plt.close()
    
    logger.info(f"Saved optimization visualizations to {save_path}")


def optimize_conservation_resources(deforestation_risk, budget_constraint, region_names=None):
    """
    Optimize allocation of conservation resources based on deforestation risk.
    
    This connects the vision-based deforestation detection with optimization.
    
    Args:
        deforestation_risk (np.ndarray): Risk scores for different regions (higher = more risk)
        budget_constraint (float): Total budget for conservation efforts
        region_names (list, optional): Names of the regions
        
    Returns:
        dict: Optimization results
    """
    logger.info("Starting conservation resource optimization")
    
    n_regions = len(deforestation_risk)
    
    if region_names is None:
        region_names = [f'Region_{i}' for i in range(n_regions)]
    
    # Create PuLP optimization model
    model = pulp.LpProblem("Conservation_Resource_Allocation", pulp.LpMaximize)
    
    # Decision variables: amount of budget allocated to each region
    allocations = [pulp.LpVariable(f"Allocation_{i}", lowBound=0) for i in range(n_regions)]
    
    # Effectiveness function: diminishing returns (square root)
    # Higher risk regions benefit more from resources
    effectiveness = []
    for i in range(n_regions):
        # Scaling factor based on risk
        risk_factor = deforestation_risk[i]
        
        # We'll use a piecewise linear approximation of diminishing returns
        # by creating breakpoints and slopes
        n_segments = 5
        max_allocation = budget_constraint  # Maximum possible allocation to one region
        segment_size = max_allocation / n_segments
        
        # Create segments
        segments = []
        for j in range(n_segments):
            # Decision variable for amount allocated in this segment
            segment_var = pulp.LpVariable(f"Segment_{i}_{j}", lowBound=0, upBound=segment_size)
            segments.append(segment_var)
            
            # Effectiveness coefficient for this segment
            # Higher coefficients for earlier segments (diminishing returns)
            coefficient = risk_factor * (1.0 - 0.15 * j)
            effectiveness.append(coefficient * segment_var)
            
        # Constraint: Total allocation to region i is sum of segment allocations
        model += pulp.lpSum(segments) == allocations[i], f"Segment_Sum_{i}"
    
    # Objective function: Maximize the effectiveness
    model += pulp.lpSum(effectiveness)
    
    # Budget constraint
    model += pulp.lpSum(allocations) <= budget_constraint, "Budget_Constraint"
    
    # Solve the model
    logger.info("Solving conservation optimization model...")
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    logger.info(f"Optimization status: {pulp.LpStatus[model.status]}")
    
    # Extract results if optimal
    if model.status == pulp.LpStatusOptimal:
        # Extract allocation values
        allocation_values = [allocation.value() for allocation in allocations]
        
        # Calculate effectiveness for each region based on allocated resources
        effectiveness_values = []
        for i in range(n_regions):
            # Approximate effectiveness function
            effectiveness_values.append(deforestation_risk[i] * np.sqrt(allocation_values[i] + 1))
        
        total_effectiveness = sum(effectiveness_values)
        
        # Create results dictionary
        results = {
            'status': 'Optimal',
            'objective_value': pulp.value(model.objective),
            'allocations': allocation_values,
            'effectiveness': effectiveness_values,
            'total_effectiveness': total_effectiveness,
            'regions': region_names
        }
        
        return results
    else:
        logger.error("Conservation optimization did not find an optimal solution")
        return {'status': 'Failed'}


def visualize_conservation_results(results, save_path=None):
    """
    Visualize conservation resource optimization results.
    
    Args:
        results (dict): Optimization results
        save_path (str, optional): Path to save visualizations
    """
    if results['status'] != 'Optimal':
        logger.error("Cannot visualize: optimization did not find an optimal solution")
        return
    
    # Set up directory for saving figures
    if save_path is None:
        save_path = RESULTS_DIR / "figures"
        os.makedirs(save_path, exist_ok=True)
    
    regions = results['regions']
    allocations = results['allocations']
    effectiveness = results['effectiveness']
    
    # 1. Plot resource allocation across regions
    plt.figure(figsize=(12, 6))
    
    # Sort regions by allocation for better visualization
    sorted_indices = np.argsort(allocations)[::-1]
    sorted_regions = [regions[i] for i in sorted_indices]
    sorted_allocations = [allocations[i] for i in sorted_indices]
    
    plt.bar(sorted_regions, sorted_allocations, color='skyblue')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Resource Allocation ($)', fontsize=12)
    plt.title('Optimal Conservation Resource Allocation', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, "conservation_allocation.png"), dpi=300)
    plt.close()
    
    # 2. Plot effectiveness across regions
    plt.figure(figsize=(12, 6))
    
    sorted_effectiveness = [effectiveness[i] for i in sorted_indices]
    
    plt.bar(sorted_regions, sorted_effectiveness, color='green')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Effectiveness Score', fontsize=12)
    plt.title('Expected Conservation Effectiveness by Region', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, "conservation_effectiveness.png"), dpi=300)
    plt.close()
    
    # 3. Create a scatter plot of allocation vs. effectiveness
    plt.figure(figsize=(8, 6))
    
    plt.scatter(allocations, effectiveness, c=range(len(regions)), cmap='viridis', 
                s=100, edgecolors='black', alpha=0.7)
    
    for i, region in enumerate(regions):
        plt.annotate(region, (allocations[i], effectiveness[i]), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center')
    
    plt.xlabel('Resource Allocation ($)', fontsize=12)
    plt.ylabel('Effectiveness Score', fontsize=12)
    plt.title('Allocation vs. Effectiveness by Region', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(label='Region Index')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, "allocation_vs_effectiveness.png"), dpi=300)
    plt.close()
    
    logger.info(f"Saved conservation optimization visualizations to {save_path}")


def main():
    """Main function to run the optimization pipeline."""
    logger.info("Starting resource optimization pipeline")
    
    # Load waste data
    waste_data = load_waste_data()
    
    # Optimize waste allocation
    waste_results = optimize_waste_allocation(waste_data)
    
    # Visualize waste optimization results
    if waste_results['status'] == 'Optimal':
        visualize_optimization_results(waste_results, waste_data)
        
        # Save optimization results
        results_dir = RESULTS_DIR / "reports"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary DataFrame
        waste_summary = pd.DataFrame({
            'Metric': ['Total Recycled Waste (tons/day)', 
                      'Environmental Benefit (CO2e saved/day)', 
                      'Total Cost ($/day)',
                      'Average Recycling Rate'],
            'Value': [sum(waste_results['total_recycled']),
                     waste_results['environmental_benefit'],
                     waste_results['total_cost'],
                     np.mean(waste_results['recycling_rate'])]
        })
        
        waste_summary.to_csv(results_dir / "waste_optimization_summary.csv", index=False)
        logger.info("Saved waste optimization results to CSV")
    
    # Example of conservation resource optimization
    logger.info("Demonstrating integration with vision-based deforestation detection")
    
    # Sample deforestation risk from vision model (higher = more risk)
    # In practice, this would come from the deforestation detection model
    n_regions = 10
    np.random.seed(42)
    deforestation_risk = np.random.uniform(0.1, 1.0, n_regions)
    region_names = [f"Forest_{chr(65+i)}" for i in range(n_regions)]
    
    # Optimize conservation resources
    conservation_results = optimize_conservation_resources(
        deforestation_risk=deforestation_risk,
        budget_constraint=1000000,  # $1M budget
        region_names=region_names
    )
    
    # Visualize conservation optimization results
    if conservation_results['status'] == 'Optimal':
        visualize_conservation_results(conservation_results)
        
        # Save conservation optimization results
        results_dir = RESULTS_DIR / "reports"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        conservation_df = pd.DataFrame({
            'Region': conservation_results['regions'],
            'Deforestation_Risk': deforestation_risk,
            'Resource_Allocation': conservation_results['allocations'],
            'Expected_Effectiveness': conservation_results['effectiveness']
        })
        
        conservation_df.to_csv(results_dir / "conservation_optimization_results.csv", index=False)
        logger.info("Saved conservation optimization results to CSV")
    
    logger.info("Resource optimization pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 