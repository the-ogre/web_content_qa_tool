import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_model_directory():
    """Check if the model directory exists and has the expected structure"""
    # Get the model directory
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    
    # Check if the directory exists
    if not os.path.exists(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        logger.info("Creating model directory...")
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Created model directory at: {model_dir}")
        return False
    
    # Required files
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "tokenizer.json"
    ]
    
    # Check if all required files exist
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            missing_files.append(file)
    
    # Check for vector linear directory
    vector_linear_dir = os.path.join(model_dir, "2_Dense_768")
    if not os.path.exists(vector_linear_dir):
        missing_files.append("2_Dense_768/")
    else:
        # Check for pytorch_model.bin in vector linear directory
        if not os.path.exists(os.path.join(vector_linear_dir, "pytorch_model.bin")):
            missing_files.append("2_Dense_768/pytorch_model.bin")
    
    if missing_files:
        logger.error(f"Missing required files in model directory: {', '.join(missing_files)}")
        logger.info(f"Please ensure all required files are present in: {model_dir}")
        return False
    
    logger.info(f"Model directory found and has all required files: {model_dir}")
    return True

if __name__ == "__main__":
    if check_model_directory():
        print("Model directory is ready to use.")
    else:
        print("Model directory is not properly set up.")
        sys.exit(1)