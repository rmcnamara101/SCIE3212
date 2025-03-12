import numpy as np
import matplotlib.pyplot as plt

# Load full sweep results
sweep_data = np.load('data/parameter_sweep/sweep_results.npz', allow_pickle=True)
ratios = sweep_data['lambda_D_ratios']
metrics = sweep_data['metrics']

# Analysis arrays
final_volumes_H = []
final_volumes_D = []
final_volumes_N = []
total_growth_rates = []
max_radii = []

# Analyze each simulation
for sim_metrics in metrics:
    # Get volumes for each cell type
    volumes_H = sim_metrics['volumes_H']
    volumes_D = sim_metrics['volumes_D']
    volumes_N = sim_metrics['volumes_N']
    
    # Store final volumes
    final_volumes_H.append(volumes_H[-1])
    final_volumes_D.append(volumes_D[-1])
    final_volumes_N.append(volumes_N[-1])
    
    # Calculate total volume and its growth rate
    total_volumes = volumes_H + volumes_D + volumes_N
    growth_rate = np.gradient(total_volumes, sim_metrics['timesteps'])
    total_growth_rates.append(np.mean(growth_rate))
    
    # Store maximum radius
    max_radii.append(np.max(sim_metrics['radii']))

# Create visualization
plt.figure(figsize=(15, 10))

# Plot 1: Final volumes by cell type
plt.subplot(221)
plt.plot(ratios, final_volumes_H, 'b-', label='Healthy')
plt.plot(ratios, final_volumes_D, 'r-', label='Diseased')
plt.plot(ratios, final_volumes_N, 'k-', label='Necrotic')
plt.xlabel('λ_D/λ_H Ratio')
plt.ylabel('Final Volume')
plt.title('Final Volumes by Cell Type')
plt.legend()

# Plot 2: Composition as percentage
plt.subplot(222)
total_volumes = np.array(final_volumes_H) + np.array(final_volumes_D) + np.array(final_volumes_N)
plt.stackplot(ratios, 
             [100 * np.array(final_volumes_H) / total_volumes,
              100 * np.array(final_volumes_D) / total_volumes,
              100 * np.array(final_volumes_N) / total_volumes],
             labels=['Healthy', 'Diseased', 'Necrotic'])
plt.xlabel('λ_D/λ_H Ratio')
plt.ylabel('Composition (%)')
plt.title('Tumor Composition')
plt.legend()

# Plot 3: Growth rates
plt.subplot(223)
plt.plot(ratios, total_growth_rates)
plt.xlabel('λ_D/λ_H Ratio')
plt.ylabel('Average Growth Rate')
plt.title('Total Tumor Growth Rate')

# Plot 4: Max radius
plt.subplot(224)
plt.plot(ratios, max_radii)
plt.xlabel('λ_D/λ_H Ratio')
plt.ylabel('Maximum Radius')
plt.title('Maximum Tumor Radius')

plt.tight_layout()
plt.show()