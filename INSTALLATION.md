# Installation Guide

This document provides detailed instructions for setting up the AI for Sustainable Ecosystems project environment.

## System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)
- **RAM**: Minimum 8GB, recommended 16GB+ for training models
- **Storage**: At least 50GB free space (datasets can be large)
- **GPU**: Recommended for training deep learning models (NVIDIA GPU with CUDA support)

## Prerequisites

Before starting, ensure you have the following installed:

1. **Python**: Version 3.8 or higher
   - [Download Python](https://www.python.org/downloads/)
   - Verify installation with: `python --version`

2. **Git**: For version control
   - [Download Git](https://git-scm.com/downloads)
   - Verify installation with: `git --version`

3. **Package Manager**: pip (comes with Python) or conda
   - If using conda: [Download Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - Verify pip installation with: `pip --version`

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-sustainable-ecosystems.git
cd ai-sustainable-ecosystems
```

### 2. Set Up Virtual Environment

#### Using venv (Python's built-in virtual environment)

```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# On Windows
env\Scripts\activate

# On macOS/Linux
source env/bin/activate
```

#### Using conda (Alternative)

```bash
# Create conda environment
conda create -n ecosystems python=3.9

# Activate conda environment
conda activate ecosystems
```

### 3. Install Dependencies

```bash
# Install dependencies using pip
pip install -r requirements.txt
```

#### Potential Issues and Solutions

- **GDAL Installation Issues**: GDAL can be challenging to install on some systems
  - On Windows: `pip install GDAL-<version>-cp<python-version>-cp<python-version>-win_amd64.whl` (download appropriate wheel from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal))
  - On macOS: `brew install gdal` then `pip install gdal`
  - On Ubuntu: `sudo apt-get install libgdal-dev` then `pip install gdal==<version>`

- **TensorFlow/PyTorch with GPU Support**:
  - For TensorFlow: Follow [TensorFlow GPU installation guide](https://www.tensorflow.org/install/gpu)
  - For PyTorch: Follow [PyTorch installation guide](https://pytorch.org/get-started/locally/)

### 4. Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
# Create .env file
touch .env  # On macOS/Linux
# or
echo. > .env  # On Windows
```

Add the following to the `.env` file:

```
# API Keys and Paths
GFW_API_KEY=your_gfw_api_key
EARTHENGINE_TOKEN_PATH=/path/to/your/earthengine-token

# Data Paths
DATA_DIR=./data
MODELS_DIR=./models
RESULTS_DIR=./results

# Other Configuration
USE_GPU=True  # Set to False if no GPU available
```

### 5. Download Datasets

The project uses several large datasets. We've created scripts to help you download them:

```bash
# Create data directories
python -m src.utils.setup_dirs

# Download datasets
python -m src.utils.download_data
```

Alternatively, you can manually download the datasets:

1. **Global Forest Watch (GFW) Data**:
   - Register at [Global Forest Watch](https://www.globalforestwatch.org/)
   - Access and download the data through their API or direct download

2. **iNaturalist Open Dataset**:
   - Download from [iNaturalist GitHub](https://github.com/inaturalist/inaturalist-open-data)
   
3. **Climate Data**:
   - Download from [Berkeley Earth](http://berkeleyearth.org/data/)

4. **Waste & Recycling Statistics**:
   - Download from [World Bank](https://datatopics.worldbank.org/what-a-waste/) or [Eurostat](https://ec.europa.eu/eurostat/web/circular-economy/information-data)

Place all downloaded datasets in their respective directories under the `data/` folder.

### 6. Verify Installation

Run the following to verify that everything is set up correctly:

```bash
# Run tests
pytest tests/

# Run a simple model to verify functionality
python -m src.utils.verify_setup
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named X**
   - Solution: Ensure you activated the virtual environment and installed all dependencies: `pip install -r requirements.txt`

2. **CUDA/GPU issues**
   - Solution: Verify your CUDA version and ensure it's compatible with installed TensorFlow/PyTorch versions

3. **Memory Errors**
   - Solution: Reduce batch sizes in configuration files or use data sampling for development

4. **Permission Errors**
   - Solution: Ensure you have appropriate permissions for the directories you're using

### Getting Help

If you encounter issues not covered here:

1. Check the [project issues page](https://github.com/your-username/ai-sustainable-ecosystems/issues)
2. Create a new issue with detailed information about your problem
3. Consult the documentation for specific libraries causing issues

## Next Steps

Once installation is complete, refer to the main [README.md](./README.md) for usage instructions and project details.

## Updating

To update the project with the latest changes:

```bash
git pull origin main
pip install -r requirements.txt  # Install any new dependencies
``` 