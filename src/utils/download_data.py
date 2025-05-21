#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset downloader for AI for Sustainable Ecosystems project.
This script downloads and sets up the required datasets.
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
import logging
import shutil
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("dataset_downloader")

# Default data directory
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "./data")

# Dataset URLs - replace with actual URLs when available
DATASETS = {
    "gfw": {
        "description": "Global Forest Watch data",
        "urls": {
            # Sample URLs - these should be replaced with actual download links
            "forest_cover": "https://example.com/gfw/forest_cover.tif",
            "deforestation": "https://example.com/gfw/deforestation.zip",
        },
        "dir": "gfw"
    },
    "inaturalist": {
        "description": "iNaturalist Open Dataset",
        "urls": {
            "sample": "https://example.com/inaturalist/sample.zip",
        },
        "dir": "inaturalist"
    },
    "climate": {
        "description": "Berkeley Earth Climate Data",
        "urls": {
            "global_temp": "https://example.com/berkeley/global_temp.csv",
        },
        "dir": "climate"
    },
    "waste": {
        "description": "Waste & Recycling Statistics",
        "urls": {
            "what_a_waste": "https://example.com/worldbank/what_a_waste.csv",
        },
        "dir": "waste"
    }
}


def download_file(url, destination, chunk_size=8192):
    """
    Download a file from a URL to a destination path with progress bar.
    
    Args:
        url (str): URL to download
        destination (str): Path to save the downloaded file
        chunk_size (int): Size of chunks for download streaming
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get the total file size in bytes
        total_size = int(response.headers.get('content-length', 0))
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress bar
        with open(destination, 'wb') as f, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                bar.update(size)
        
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False


def extract_archive(archive_path, extract_dir):
    """
    Extract a ZIP or TAR archive.
    
    Args:
        archive_path (str): Path to the archive file
        extract_dir (str): Directory to extract the archive to
    
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        logger.info(f"Extracting {archive_path} to {extract_dir}")
        
        # Create extraction directory if it doesn't exist
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract based on file extension
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        
        elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(archive_path) as tar_ref:
                tar_ref.extractall(extract_dir)
        
        else:
            logger.warning(f"Unsupported archive format: {archive_path}")
            return False
        
        logger.info(f"Successfully extracted {archive_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error extracting {archive_path}: {str(e)}")
        return False


def download_dataset(dataset_name, data_dir):
    """
    Download a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset to download
        data_dir (str): Base directory to store the dataset
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    if dataset_name not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False
    
    dataset = DATASETS[dataset_name]
    dataset_dir = os.path.join(data_dir, dataset["dir"])
    os.makedirs(dataset_dir, exist_ok=True)
    
    logger.info(f"Downloading {dataset['description']} to {dataset_dir}")
    
    success = True
    for name, url in dataset["urls"].items():
        file_name = url.split('/')[-1]
        file_path = os.path.join(dataset_dir, file_name)
        
        if os.path.exists(file_path):
            logger.info(f"File already exists: {file_path}")
            continue
        
        logger.info(f"Downloading {name} from {url}")
        if download_file(url, file_path):
            # Extract archives if needed
            if file_path.endswith(('.zip', '.tar', '.tar.gz', '.tgz')):
                extract_dir = os.path.join(dataset_dir, name)
                if not extract_archive(file_path, extract_dir):
                    success = False
        else:
            success = False
    
    return success


def download_all_datasets(data_dir):
    """
    Download all datasets.
    
    Args:
        data_dir (str): Base directory to store the datasets
    
    Returns:
        bool: True if all downloads were successful, False otherwise
    """
    success = True
    for dataset_name in DATASETS:
        if not download_dataset(dataset_name, data_dir):
            success = False
    
    return success


def setup_dirs(data_dir):
    """
    Set up the directory structure for datasets.
    
    Args:
        data_dir (str): Base directory for datasets
    """
    for dataset_name, dataset_info in DATASETS.items():
        dataset_dir = os.path.join(data_dir, dataset_info["dir"])
        os.makedirs(dataset_dir, exist_ok=True)
        logger.info(f"Created directory: {dataset_dir}")


def main():
    parser = argparse.ArgumentParser(description='Download datasets for AI for Sustainable Ecosystems project')
    parser.add_argument('--dataset', type=str, help='Specific dataset to download',
                        choices=list(DATASETS.keys()) + ['all'], default='all')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Directory to store datasets (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--setup-only', action='store_true',
                        help='Only set up directories without downloading data')
    args = parser.parse_args()
    
    # Create base data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.setup_only:
        setup_dirs(args.data_dir)
        logger.info("Directory setup complete.")
        return 0
    
    if args.dataset == 'all':
        success = download_all_datasets(args.data_dir)
    else:
        success = download_dataset(args.dataset, args.data_dir)
    
    if success:
        logger.info("All downloads completed successfully.")
        return 0
    else:
        logger.warning("Some downloads failed. Check the log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 