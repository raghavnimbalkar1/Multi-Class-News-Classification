"""
Configuration Loader Utility
Loads project configuration from config.yaml and provides helper functions
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Loads and manages project configuration."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern - ensure only one instance."""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize config loader."""
        if self._config is None:
            self.load_config()
    
    def load_config(self):
        """Load configuration from config.yaml file."""
        # Find project root by looking for config.yaml
        current_dir = Path(__file__).parent.parent  # Go up from src/ to project root
        config_path = current_dir / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        self.project_root = current_dir
    
    def get(self, key: str, default=None) -> Any:
        """Get config value by dot-notation key (e.g., 'paths.data.raw')."""
        if self._config is None:
            self.load_config()
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_path(self, key: str) -> str:
        """Get a file/directory path from config and convert to absolute path."""
        relative_path = self.get(key)
        if relative_path is None:
            raise ValueError(f"Path config '{key}' not found")
        
        absolute_path = self.project_root / relative_path
        return str(absolute_path)
    
    def ensure_dir_exists(self, key: str) -> str:
        """Get a path from config, ensure directory exists, return absolute path."""
        absolute_path = self.get_path(key)
        os.makedirs(absolute_path, exist_ok=True)
        return absolute_path
    
    def create_file_dir(self, key: str) -> str:
        """Get a file path and ensure parent directory exists."""
        absolute_path = self.get_path(key)
        parent_dir = os.path.dirname(absolute_path)
        os.makedirs(parent_dir, exist_ok=True)
        return absolute_path
    
    def get_full_config(self) -> Dict:
        """Get the entire configuration dictionary."""
        if self._config is None:
            self.load_config()
        return self._config


# Global config instance
config = ConfigLoader()


def get_config_path(key: str) -> str:
    """Convenience function to get a config path."""
    return config.get_path(key)


def get_config_value(key: str, default=None):
    """Convenience function to get a config value."""
    return config.get(key, default)


def ensure_dir(key: str) -> str:
    """Convenience function to ensure directory exists."""
    return config.ensure_dir_exists(key)


def create_file_path(key: str) -> str:
    """Convenience function to create file path with parent directories."""
    return config.create_file_dir(key)
