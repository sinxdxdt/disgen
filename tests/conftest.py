"""Test configuration file for pytest."""

import sys
from pathlib import Path

# Add disgen to path
disgen_path = Path(__file__).parent.parent
sys.path.insert(0, str(disgen_path))
