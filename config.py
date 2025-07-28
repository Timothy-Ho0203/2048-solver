#!/usr/bin/env python3
"""
Simple configuration for model save paths.
Uses python-dotenv for .env file loading.
"""

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class Config:
    """Configuration class for model save paths."""

    MODEL_SAVE_PATH: str = 'models/trained_models'

    @classmethod
    def load_from_env_file(cls, env_file: str = '.env'):
        """Load configuration from a .env file using python-dotenv."""
        load_dotenv(env_file)
        cls.MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', cls.MODEL_SAVE_PATH)
    
    @classmethod
    def ensure_model_directory(cls, path: Optional[str] = None) -> str:
        """Ensure the model directory exists and return the path."""
        model_path = path or cls.MODEL_SAVE_PATH
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(model_path) if '/' in model_path else '.'
        if directory and directory != '.' and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        return model_path


# Load configuration from .env file on import
Config.load_from_env_file() 