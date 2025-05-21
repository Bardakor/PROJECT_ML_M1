#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Directory setup utility for AI for Sustainable Ecosystems project.
This script creates the necessary directory structure for the project.
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("setup_dirs.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("setup_dirs")

# Default directories
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "./data")
DEFAULT_MODELS_DIR = os.getenv("MODELS_DIR", "./models")
DEFAULT_RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")

# Project directory structure
PROJECT_STRUCTURE = {
    "data": {
        "gfw": {},            # Global Forest Watch data
        "inaturalist": {},    # iNaturalist wildlife image data
        "climate": {},        # Climate data
        "waste": {}           # Waste and recycling data
    },
    "notebooks": {
        "vision_tasks": {},   # Notebooks for vision tasks
        "analytical_tasks": {} # Notebooks for analytical tasks
    },
    "src": {
        "vision": {
            "deforestation": {},  # Deforestation detection models
            "wildlife": {}        # Wildlife classification models
        },
        "analytics": {
            "forecasting": {},    # Climate and trend forecasting
            "optimization": {}    # Optimization for sustainability
        },
        "utils": {}              # Utility scripts
    },
    "models": {                  # Saved model files
        "vision": {},
        "analytics": {}
    },
    "results": {                 # Output results and visualizations
        "figures": {},
        "reports": {}
    },
    "tests": {}                  # Unit tests
}


def create_directory_structure(base_dir, structure):
    """
    Recursively create a directory structure based on the provided dictionary.
    
    Args:
        base_dir (str): Base directory path
        structure (dict): Dictionary representing the directory structure
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    logger.info(f"Created directory: {base_dir}")
    
    # Create subdirectories
    for dir_name, substructure in structure.items():
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
        
        # Recursively create substructure if it exists
        if substructure:
            create_directory_structure(dir_path, substructure)


def create_init_files(directory):
    """
    Recursively create __init__.py files in all Python package directories.
    
    Args:
        directory (str): Directory path to start from
    """
    # Skip if not a directory or is a hidden directory
    if not os.path.isdir(directory) or os.path.basename(directory).startswith('.'):
        return
    
    # Create __init__.py if it's a Python package directory
    if any(os.path.isdir(os.path.join(directory, d)) for d in os.listdir(directory)):
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                pass  # Create an empty file
            logger.info(f"Created __init__.py in {directory}")
    
    # Recursively process subdirectories
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            create_init_files(item_path)


def create_env_file(project_dir):
    """
    Create a template .env file if it doesn't exist.
    
    Args:
        project_dir (str): Project root directory
    """
    env_file = os.path.join(project_dir, ".env")
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write("# Environment variables for AI for Sustainable Ecosystems project\n")
            f.write("\n# API Keys\n")
            f.write("GFW_API_KEY=your_gfw_api_key_here\n")
            f.write("EARTHENGINE_TOKEN_PATH=/path/to/your/earthengine-token\n")
            f.write("\n# Data Paths\n")
            f.write("DATA_DIR=./data\n")
            f.write("MODELS_DIR=./models\n")
            f.write("RESULTS_DIR=./results\n")
            f.write("\n# Model Configuration\n")
            f.write("USE_GPU=True  # Set to False if no GPU available\n")
        logger.info(f"Created template .env file at {env_file}")


def create_gitignore(project_dir):
    """
    Create a .gitignore file if it doesn't exist.
    
    Args:
        project_dir (str): Project root directory
    """
    gitignore_file = os.path.join(project_dir, ".gitignore")
    if not os.path.exists(gitignore_file):
        with open(gitignore_file, "w") as f:
            f.write("# Python\n")
            f.write("__pycache__/\n")
            f.write("*.py[cod]\n")
            f.write("*$py.class\n")
            f.write("*.so\n")
            f.write(".Python\n")
            f.write("env/\n")
            f.write("venv/\n")
            f.write("build/\n")
            f.write("develop-eggs/\n")
            f.write("dist/\n")
            f.write("downloads/\n")
            f.write("eggs/\n")
            f.write(".eggs/\n")
            f.write("*.egg-info/\n")
            f.write(".installed.cfg\n")
            f.write("*.egg\n")
            f.write("\n# Jupyter Notebook\n")
            f.write(".ipynb_checkpoints\n")
            f.write("\n# Environment\n")
            f.write(".env\n")
            f.write(".venv\n")
            f.write("\n# Data files (optional - may want to keep smaller datasets)\n")
            f.write("data/*/raw/*\n")
            f.write("*.zip\n")
            f.write("*.tar.gz\n")
            f.write("\n# Model files (optional - may want to keep smaller models)\n")
            f.write("models/*/*.h5\n")
            f.write("models/*/*.pkl\n")
            f.write("models/*/*.pt\n")
            f.write("\n# Logs\n")
            f.write("*.log\n")
            f.write("\n# OS specific\n")
            f.write(".DS_Store\n")
            f.write("Thumbs.db\n")
        logger.info(f"Created .gitignore file at {gitignore_file}")


def main():
    parser = argparse.ArgumentParser(description='Set up directory structure for AI for Sustainable Ecosystems project')
    parser.add_argument('--project-dir', type=str, default=".",
                        help='Project root directory (default: current directory)')
    parser.add_argument('--create-init', action='store_true',
                        help='Create __init__.py files in all Python package directories')
    parser.add_argument('--create-env', action='store_true',
                        help='Create a template .env file')
    parser.add_argument('--create-gitignore', action='store_true',
                        help='Create a .gitignore file')
    args = parser.parse_args()
    
    # Create directory structure
    create_directory_structure(args.project_dir, PROJECT_STRUCTURE)
    logger.info("Directory structure created successfully")
    
    # Create __init__.py files if requested
    if args.create_init:
        create_init_files(os.path.join(args.project_dir, "src"))
        logger.info("Created __init__.py files in all Python package directories")
    
    # Create .env file if requested
    if args.create_env:
        create_env_file(args.project_dir)
    
    # Create .gitignore file if requested
    if args.create_gitignore:
        create_gitignore(args.project_dir)
    
    logger.info("Setup complete")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 