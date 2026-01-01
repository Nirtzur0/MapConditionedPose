
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.theme import Theme

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Global rich console with custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
})
console = Console(theme=custom_theme)


def setup_logging(level=logging.INFO, name: str = "pipeline"):
    """
    Set up logging for the entire application.
    This function configures the root logger to use RichHandler for
    beautiful console output and a file handler to save logs to a file.
    """
    log_path = Path(f'logs/{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []  # Clear existing handlers

    # Create a RichHandler for console output
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=False,
        show_level=False,
        show_path=False,
        markup=True,
    )
    console_handler.setLevel(level)

    # Create a FileHandler for file output
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)

    # Create a formatter for FILE output only (keep it detailed)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    # NOTE: We do NOT set a formatter for console_handler, allowing Rich to use its modern, concise defaults.

    # Add the handlers to the root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Silence noisy third-party libraries
    noisy_loggers = [
        "matplotlib", "PIL", "numba", "h5py", "absl", 
        "fiona", "shapely", "rasterio", "tensorflow", "torch",
        "pytorch_lightning", "urllib3", "asyncio"
    ]
    for lib in noisy_loggers:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Suppress TensorFlow warnings
    import os
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    console.print(f"[dim]Log file: {log_path}[/dim]")

    return logging.getLogger("rich")


def print_section(title: str, subtitle: str = ""):
    """Print a clean, visually appealing section header"""
    panel_content = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        panel_content += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(panel_content, expand=False, border_style="cyan"))


def print_success(message: str):
    """Print a success message with checkmark"""
    console.print(f"[success]✓[/success] {message}")


def print_info(message: str):
    """Print an info message"""
    console.print(f"[info]→[/info] {message}")


def print_warning(message: str):
    """Print a warning message"""
    console.print(f"[warning]⚠[/warning] {message}")


def print_error(message: str):
    """Print an error message"""
    console.print(f"[error]✗[/error] {message}")


def create_progress_bar():
    """Create a rich progress bar for long-running operations"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )


if __name__ == "__main__":
    # Example usage:
    setup_logging()
    
    logger = logging.getLogger("my_app")
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    try:
        x = 1 / 0
    except ZeroDivisionError:
        logger.exception("This is an exception message.")

