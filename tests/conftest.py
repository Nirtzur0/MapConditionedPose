import sys
import os
from pathlib import Path

# Add project root to sys.path
# This ensures that 'src' is importable from tests
root_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_dir))
