#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verification script to check if the project environment is set up correctly.
This script checks dependencies, directory structure, and environment variables.
"""

import os
import sys
import logging
import importlib
import platform
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("verify_setup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("verify_setup")

# Required dependencies
CORE_DEPENDENCIES = [
    "numpy", "pandas", "scipy", "matplotlib", "scikit-learn"
]
VISION_DEPENDENCIES = [
    "torch", "tensorflow", "opencv-python", "scikit-image"
]
GEO_DEPENDENCIES = [
    "geopandas", "rasterio"
]
DATA_DEPENDENCIES = [
    "tqdm", "requests", "pyarrow", "dask"
]

# Required directories
REQUIRED_DIRS = [
    "data", "notebooks", "src", "models", "results", "tests"
]
DATA_SUBDIRS = [
    "gfw", "inaturalist", "climate", "waste"
]

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    min_version = (3, 8)
    if version >= min_version:
        logger.info(f"{GREEN}Python version {version.major}.{version.minor}.{version.micro} OK{RESET}")
        return True
    else:
        logger.error(f"{RED}Python version {version.major}.{version.minor}.{version.micro} is too old. "
                     f"Minimum required version is {min_version[0]}.{min_version[1]}{RESET}")
        return False


def check_dependency(dependency):
    """Check if a dependency is installed."""
    try:
        # Handle special cases for packages with different import names
        if dependency == "opencv-python":
            importlib.import_module("cv2")
        else:
            importlib.import_module(dependency.split(">=")[0])
        return True
    except ImportError:
        return False


def check_dependencies(dependencies, description):
    """Check if multiple dependencies are installed."""
    logger.info(f"Checking {description} dependencies...")
    all_installed = True
    
    for dependency in dependencies:
        if check_dependency(dependency):
            logger.info(f"{GREEN}✓ {dependency} installed{RESET}")
        else:
            logger.error(f"{RED}✗ {dependency} not installed{RESET}")
            all_installed = False
    
    return all_installed


def check_gpu_support():
    """Check if GPU support is available for deep learning."""
    gpu_available = False
    
    # Check TensorFlow GPU support
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            gpu_available = True
            logger.info(f"{GREEN}✓ TensorFlow GPU support available: {len(gpus)} GPU(s){RESET}")
            for gpu in gpus:
                logger.info(f"  - {gpu.name}")
        else:
            logger.warning(f"{YELLOW}⚠ TensorFlow GPU support not available{RESET}")
    except Exception as e:
        logger.warning(f"{YELLOW}⚠ Could not check TensorFlow GPU support: {str(e)}{RESET}")
    
    # Check PyTorch GPU support
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            logger.info(f"{GREEN}✓ PyTorch GPU support available: {torch.cuda.device_count()} GPU(s){RESET}")
            logger.info(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning(f"{YELLOW}⚠ PyTorch GPU support not available{RESET}")
    except Exception as e:
        logger.warning(f"{YELLOW}⚠ Could not check PyTorch GPU support: {str(e)}{RESET}")
    
    return gpu_available


def check_directory_structure():
    """Check if the project directory structure is set up correctly."""
    logger.info("Checking project directory structure...")
    
    # Check main directories
    all_dirs_exist = True
    for dir_name in REQUIRED_DIRS:
        if os.path.isdir(dir_name):
            logger.info(f"{GREEN}✓ {dir_name}/ directory exists{RESET}")
        else:
            logger.error(f"{RED}✗ {dir_name}/ directory not found{RESET}")
            all_dirs_exist = False
    
    # Check data subdirectories if data directory exists
    if os.path.isdir("data"):
        for subdir in DATA_SUBDIRS:
            subdir_path = os.path.join("data", subdir)
            if os.path.isdir(subdir_path):
                logger.info(f"{GREEN}✓ {subdir_path}/ directory exists{RESET}")
            else:
                logger.warning(f"{YELLOW}⚠ {subdir_path}/ directory not found{RESET}")
    
    return all_dirs_exist


def check_environment_variables():
    """Check if required environment variables are set."""
    logger.info("Checking environment variables...")
    
    # Load environment variables
    load_dotenv()
    
    # List of environment variables to check
    env_vars = [
        "DATA_DIR",
        "MODELS_DIR",
        "RESULTS_DIR",
        "USE_GPU"
    ]
    
    # Optional environment variables
    optional_env_vars = [
        "GFW_API_KEY",
        "EARTHENGINE_TOKEN_PATH"
    ]
    
    # Check required environment variables
    all_vars_set = True
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"{GREEN}✓ {var} is set to: {value}{RESET}")
        else:
            logger.error(f"{RED}✗ {var} is not set{RESET}")
            all_vars_set = False
    
    # Check optional environment variables
    for var in optional_env_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"{GREEN}✓ {var} is set{RESET}")
        else:
            logger.warning(f"{YELLOW}⚠ {var} is not set (optional){RESET}")
    
    return all_vars_set


def check_system_info():
    """Collect and display system information."""
    logger.info("Collecting system information...")
    
    # Get system information
    logger.info(f"Operating System: {platform.system()} {platform.version()}")
    logger.info(f"Architecture: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    
    # Get memory information
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        memory = int(line.split()[1]) // 1024  # Convert to MB
                        logger.info(f"RAM: {memory} MB")
                        break
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            memory = int(result.stdout.strip()) // (1024 * 1024)  # Convert to MB
            logger.info(f"RAM: {memory} MB")
        elif platform.system() == "Windows":
            result = subprocess.run(['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'], capture_output=True, text=True)
            memory_line = result.stdout.strip().split('\n')[1]
            memory = int(memory_line) // (1024 * 1024)  # Convert to MB
            logger.info(f"RAM: {memory} MB")
    except Exception as e:
        logger.warning(f"Could not determine RAM: {str(e)}")


def run_simple_test():
    """Run a simple test to verify the setup."""
    logger.info("Running a simple test...")
    
    try:
        # Try to import and use common libraries
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Create a simple array and dataframe
        arr = np.random.rand(10, 10)
        df = pd.DataFrame(arr)
        
        # Verify numpy operations
        arr_sum = np.sum(arr)
        logger.info(f"{GREEN}✓ NumPy operations successful{RESET}")
        
        # Verify pandas operations
        df_mean = df.mean().mean()
        logger.info(f"{GREEN}✓ Pandas operations successful{RESET}")
        
        # Try to create and save a simple plot
        plt.figure(figsize=(4, 3))
        plt.imshow(arr, cmap='viridis')
        plt.colorbar()
        plt.title('Random Matrix')
        
        # Create results directory if it doesn't exist
        os.makedirs("results/figures", exist_ok=True)
        
        # Save the figure
        plt.savefig("results/figures/test_figure.png")
        plt.close()
        logger.info(f"{GREEN}✓ Matplotlib operations successful{RESET}")
        
        logger.info(f"{GREEN}All basic library tests passed!{RESET}")
        return True
    
    except Exception as e:
        logger.error(f"{RED}Simple test failed: {str(e)}{RESET}")
        return False


def main():
    """Main verification function."""
    logger.info("Starting environment verification for AI for Sustainable Ecosystems project\n")
    
    # Check Python version
    python_ok = check_python_version()
    print()
    
    # Check dependencies
    core_ok = check_dependencies(CORE_DEPENDENCIES, "core")
    print()
    vision_ok = check_dependencies(VISION_DEPENDENCIES, "vision")
    print()
    geo_ok = check_dependencies(GEO_DEPENDENCIES, "geospatial")
    print()
    data_ok = check_dependencies(DATA_DEPENDENCIES, "data processing")
    print()
    
    # Check GPU availability
    gpu_available = check_gpu_support()
    print()
    
    # Check directory structure
    dirs_ok = check_directory_structure()
    print()
    
    # Check environment variables
    env_ok = check_environment_variables()
    print()
    
    # Get system information
    check_system_info()
    print()
    
    # Run a simple test
    test_ok = run_simple_test()
    print()
    
    # Final verification summary
    logger.info("Verification Summary:")
    
    all_passed = python_ok and core_ok and dirs_ok and test_ok
    critical_passed = all_passed
    
    # Calculate how many dependency sets passed
    dependency_sets = [core_ok, vision_ok, geo_ok, data_ok]
    passed_dependencies = sum(dependency_sets)
    total_dependencies = len(dependency_sets)
    
    # Print summary
    logger.info(f"Python Version check: {'✓' if python_ok else '✗'}")
    logger.info(f"Core Dependencies: {'✓' if core_ok else '✗'}")
    logger.info(f"Vision Dependencies: {'✓' if vision_ok else '✗'}")
    logger.info(f"Geospatial Dependencies: {'✓' if geo_ok else '✗'}")
    logger.info(f"Data Processing Dependencies: {'✓' if data_ok else '✗'}")
    logger.info(f"Directory Structure: {'✓' if dirs_ok else '✗'}")
    logger.info(f"Environment Variables: {'✓' if env_ok else '✗'}")
    logger.info(f"GPU Support: {'✓' if gpu_available else '⚠'}")
    logger.info(f"Basic Functionality Test: {'✓' if test_ok else '✗'}")
    
    # Final assessment
    if critical_passed:
        logger.info(f"\n{GREEN}Verification PASSED! The critical components are properly set up.{RESET}")
        if passed_dependencies < total_dependencies:
            logger.warning(f"\n{YELLOW}Note: {total_dependencies - passed_dependencies} dependency group(s) not fully installed. "
                         f"Some project features may not work.{RESET}")
        if not gpu_available:
            logger.warning(f"\n{YELLOW}Note: No GPU detected. Model training will be slower.{RESET}")
        if not env_ok:
            logger.warning(f"\n{YELLOW}Note: Some environment variables are not set properly.{RESET}")
    else:
        logger.error(f"\n{RED}Verification FAILED! Please fix the issues above before continuing.{RESET}")
    
    return 0 if critical_passed else 1


if __name__ == "__main__":
    sys.exit(main()) 