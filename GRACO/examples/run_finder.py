"""Run the FINDER baseline in three lines.

    python examples/run_finder.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graco.trainers import Trainer
from graco.utils.config import load_config

Trainer(load_config("finder")).train()
