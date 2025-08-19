from pathlib import Path
import sys
import yaml

if sys.platform == "darwin":
    # Point to the actual project root that contains 'src'
    proj = Path(__file__).parent
    output_dir = str(proj / "laboratory" / "saved_simulations")
else:
    # Point to the actual project root that contains 'src'
    proj = Path(__file__).parent
    output_dir = str(proj / "laboratory" / "saved_simulations")

# Prepend so it wins over anything else
sys.path.insert(0, str(proj))

from src.growkit.Simulator import TumorGrowthSimulator

# Use the YAML that actually exists under this repo
cfg = proj / "configs" / "csc-t-n.yaml"
simulator = TumorGrowthSimulator(str(cfg))

simulator.run_and_save_simulation(total_steps=20, save_interval=4, save_physics_fields=True, output_dir=output_dir)