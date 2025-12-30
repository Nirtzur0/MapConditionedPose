"""
Clean logging configuration for the project.
Suppresses third-party warnings and formats project logs cleanly.
"""
import logging
import warnings
import sys

def setup_clean_logging(level=logging.INFO):
    """
    Configure clean, concise logging for the project.
    
    Args:
        level: Logging level for project logs
    """
    # Suppress third-party warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*UnstableSpecificationWarning.*')
    
    # Configure logging format
    log_format = '%(levelname)-5s | %(name)-25s | %(message)s'
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Set high threshold for root
    
    # Configure our project loggers
    project_prefixes = [
        '__main__',
        'scene_generation',
        'data_generation', 
        'src.',
        'training',
        'pipeline'
    ]
    
    for prefix in project_prefixes:
        logger = logging.getLogger(prefix)
        logger.setLevel(level)
    
    # Silence noisy third-party loggers
    logging.getLogger('keras').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('zarr').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)
    logging.getLogger('trimesh').setLevel(logging.ERROR)
    
    # Sionna/Mitsuba uses print statements for warnings, harder to suppress
    # Could redirect stderr but risky
    
    # Configure handler if not already present
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(handler)
    
    return root_logger
