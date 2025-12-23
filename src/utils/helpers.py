"""
Utility Functions
=================

Helper functions for configuration loading, logging setup,
device detection, and other common operations.
"""

import os
import yaml
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        log_format: Log message format
    
    Returns:
        Configured root logger
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers
    )
    
    # Set transformers logging level
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    return logging.getLogger()


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
    
    Returns:
        torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    
    return device


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with memory info in GB
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "allocated": torch.cuda.memory_allocated(0) / 1024**3,
        "cached": torch.cuda.memory_reserved(0) / 1024**3,
        "free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
    }


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For reproducibility on GPU (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seed set to {seed}")


def create_output_dirs(base_dir: str = "./outputs") -> Dict[str, Path]:
    """
    Create output directories for training.
    
    Args:
        base_dir: Base output directory
    
    Returns:
        Dictionary with paths to created directories
    """
    base_path = Path(base_dir)
    
    dirs = {
        "checkpoints": base_path / "checkpoints",
        "logs": base_path / "logs",
        "predictions": base_path / "predictions"
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Created directory: {path}")
    
    return dirs


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "2h 30m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_percent": 100 * trainable / total if total > 0 else 0
    }


if __name__ == "__main__":
    # Test utilities
    logging.basicConfig(level=logging.INFO)
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test GPU info
    gpu_info = get_gpu_memory_info()
    print(f"GPU info: {gpu_info}")
    
    # Test config loading (if exists)
    try:
        config = load_config()
        print(f"Loaded config with keys: {list(config.keys())}")
    except FileNotFoundError:
        print("Config file not found (expected)")
