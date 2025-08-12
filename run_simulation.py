from pathlib import Path
import sys

# Point to the actual project root that contains 'src'
proj = Path(r"C:\Users\riley.mcnamara\Documents\code\SCIE3212")
assert (proj / "src").exists(), f"No 'src' at {proj}"

# Prepend so it wins over anything else
sys.path.insert(0, str(proj))

from src.growkit.Simulator import TumorGrowthSimulator

# Use the YAML that actually exists under this repo
cfg = proj / "templates" / "og.yaml"
simulator = TumorGrowthSimulator(str(cfg))
simulator.run_and_save_simulation(total_steps=1, save_interval=1, save_physics_fields=True, output_dir=r"C:\Users\riley.mcnamara\Documents\code\SCIE3212")