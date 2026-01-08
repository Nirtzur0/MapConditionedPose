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
    # Suppress third-party warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', module='optuna') # Optuna experimental warnings
    warnings.filterwarnings('ignore', message='.*UnstableSpecificationWarning.*')
    warnings.filterwarnings('ignore', message='.*Visual Studio Code.*')
    warnings.filterwarnings('ignore', message='.*Leftover.*')
    warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*') # PyTorch Transformer warning
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    warnings.filterwarnings('ignore', message='invalid value encountered in divide')
    warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

    # Try to silence Mitsuba (Sionna backend) C++ warnings
    try:
        import mitsuba as mi
        mi.set_log_level(mi.LogLevel.Error)
    except ImportError:
        pass
    
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
    logging.getLogger('zarr').setLevel(logging.ERROR)  # Legacy Zarr (deprecated)
    logging.getLogger('lmdb').setLevel(logging.WARNING)  # LMDB (primary storage)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)
    logging.getLogger('trimesh').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('comet_ml').setLevel(logging.ERROR)
    logging.getLogger('fiona').setLevel(logging.ERROR)
    logging.getLogger('shapely').setLevel(logging.ERROR)
    
    # Sionna/Mitsuba uses print statements for warnings, harder to suppress
    # Could redirect stderr but risky
    
    # Configure handler if not already present
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(handler)
    
    return root_logger
