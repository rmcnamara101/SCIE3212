#!/usr/bin/env python3
"""
Simple launcher script for the Simulation Analysis GUI
"""

import sys
import os
from pathlib import Path

# Add project path to sys.path
if sys.platform == "darwin":
    proj = "/Users/rileymcnamara/CODE/2025/silicokit/"
    sys.path.insert(0, proj)
else:
    proj = "C:/Users/riley.mcnamara/Documents/code/silicokit/"
    sys.path.insert(0, proj)

# Import and run the GUI
try:
    from simulation_gui import main
    print("Starting Simulation Analysis GUI...")
    main()
except ImportError as e:
    print(f"Error importing GUI: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error running GUI: {e}")
    sys.exit(1)
