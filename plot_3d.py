import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle

# Try to import marching cubes for isosurfaces
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    try:
        from scipy.spatial import ConvexHull
        HAS_SKIMAGE = False
        print("Warning: skimage not available, using alternative methods")
    except ImportError:
        HAS_SKIMAGE = False
        print("Warning: Neither skimage nor scipy.spatial available")

if sys.platform == "darwin":
    proj = "/Users/rileymcnamara/CODE/2025/silicokit/"
    sys.path.insert(0, proj)
else:
    proj = "C:/Users/riley.mcnamara/Documents/code/silicokit/"
    sys.path.insert(0, proj)

from src.growkit.PlotEngine.CellFieldPlotter import CellFieldPlotter

# Load the raw npz file
raw_data = np.load(proj + "laboratory/saved_simulations/simulation_data.npz")

# Convert npz to dictionary format
simulation_data = CellFieldPlotter._convert_npz_to_dict(raw_data)

# Create plotter from converted simulation data
plotter = CellFieldPlotter.from_simulation_data(simulation_data, step_idx=0)

# Print available populations
print("Available populations:")
for i, label in enumerate(plotter.labels):
    print(f"  Index {i}: {label}")

# Plot 3D tumor field
#plotter.plot_3d_tumor_field(simulation_data, step_idx=9, isosurface_level=0.1, cmap="PiYG")

# Get data
step_idx = 16
phi_hat = simulation_data["field_data"]["phi_hat"][step_idx]
nx, ny, nz = plotter.grid
x = np.arange(nx) * plotter.dx
y = np.arange(ny) * plotter.dx
z = np.arange(nz) * plotter.dx

# Combined isosurfaces with improved styling
if HAS_SKIMAGE:
    print("\n=== Creating Combined Isosurfaces ===")
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Better color scheme - more scientific/medical
    # Using distinct, professional colors
    colors_list = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green (more vibrant)
    edge_colors = ['#C0392B', '#2980B9', '#27AE60']  # Darker edges for definition
    alphas = [0.5, 0.5, 0.7]  # More opaque for better visibility
    thresholds = [0.15, 0.15, 0.01]
    
    legend_handles = []
    
    for i, label in enumerate(plotter.labels):
        population_density = phi_hat[i, :, :, :]
        threshold = thresholds[i]
        
        try:
            verts, faces, normals, values = measure.marching_cubes(
                population_density, threshold, spacing=(plotter.dx, plotter.dx, plotter.dx)
            )
            
            # Create mesh with edges for definition
            # Calculate face colors based on normals for better shading effect
            # Use normals to create a lighting effect
            face_colors = []
            for face in faces:
                # Get normal for this face
                face_normal = normals[face[0]]
                # Simple lighting calculation (dot product with light direction)
                light_dir = np.array([0.5, 0.5, 1.0])
                light_dir = light_dir / np.linalg.norm(light_dir)
                intensity = max(0.3, np.dot(face_normal, light_dir))
                # Create shaded color
                base_color = np.array(plt.cm.colors.to_rgb(colors_list[i]))
                shaded_color = base_color * intensity
                face_colors.append(shaded_color)
            
            # Create mesh with per-face colors for shading
            mesh = Poly3DCollection(verts[faces], 
                                   facecolors=face_colors,
                                   edgecolor=edge_colors[i], 
                                   linewidth=0.4,  # Thin edges for definition
                                   alpha=alphas[i],
                                   label=label)
            ax.add_collection3d(mesh)
            
            # Create custom legend handle
            legend_handles.append(Rectangle((0, 0), 1, 1, 
                                           facecolor=colors_list[i], 
                                           edgecolor=edge_colors[i],
                                           linewidth=2, 
                                           alpha=alphas[i], 
                                           label=label))
            print(f"  Added {label} isosurface ({len(verts)} vertices)")
        except Exception as e:
            print(f"  {label}: Error: {e}")
    
    # Set limits with some padding
    padding = 5
    ax.set_xlim(0 - padding, x.max() + padding)
    ax.set_ylim(0 - padding, y.max() + padding)
    ax.set_zlim(0 - padding, z.max() + padding)
    
    # Better axis labels
    ax.set_xlabel('X (μm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (μm)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (μm)', fontsize=12, fontweight='bold')
    
    # Improve title
    step = simulation_data["metadata"]["saved_steps"][step_idx]
    time = simulation_data["metadata"]["saved_times"][step_idx]
    ax.set_title(f'3D Cell Population Isosurfaces\nStep {step} (t={time:.2f})', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Better legend
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', fontsize=11, 
                 framealpha=0.9, fancybox=True, shadow=True)
    
    # Set background color
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set viewing angle for better perspective
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
else:
    print("Error: skimage.measure.marching_cubes is required for isosurface visualization")

# Plot 3D quadratic potential with tumor overlay
#plotter.plot_3d_quadratic_potential(simulation_data, step_idx=0, show_tumor_field=True)

# Analyze coagulation dynamics over multiple steps
#plotter.plot_coagulation_analysis(simulation_data, step_indices=[0, 1, 2, 3, 4])

