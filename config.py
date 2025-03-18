import os
import logging
from pathlib import Path

class SearchConfig:
    def __init__(self):
        # Model directory is in the same folder as the script
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        self.vector_dim = 1024  # Typical embedding dimension, adjust if your model differs
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger