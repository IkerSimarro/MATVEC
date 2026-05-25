"""Add the project root to sys.path so tests can import top-level modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
