import matplotlib.pyplot as plt
import numpy as np

def animate_tumor_slices(self, steps=100, slice_pos=None, interval=100):
    """
    Create an animation of contour plots showing tumor growth through x-axis slices.
    
    Parameters:
    -----------
    steps : int
        Number of frames in the animation
    slice_pos : int, optional
        Position of x-slice. If None, uses middle of grid
    interval : int
        Delay between frames in milliseconds
    """
    import matplotlib.animation as animation
    
    # Set default slice position to middle of grid if not specified
    if slice_pos is None:
        slice_pos = self.grid_shape[0] // 2
    
    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Tumor Growth Animation (X-Slice at {slice_pos})', fontsize=16)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Dictionary of cell types and their properties
    cell_types = {
        'Stem Cells': (self.C_S, 'Reds'),
        'Progenitor Cells': (self.C_P, 'Blues'),
        'Differentiated Cells': (self.C_D, 'Greens'),
        'Necrotic Cells': (self.C_N, 'Greys')
    }
    
    # Initialize contour plots
    contours = []
    for ax, (title, (field, cmap)) in zip(axes, cell_types.items()):
        # Create initial contour plot
        cont = ax.contourf(
            np.zeros((self.grid_shape[1], self.grid_shape[2])),
            levels=np.linspace(0, 1, 20),
            cmap=cmap
        )
        ax.set_title(title)
        fig.colorbar(cont, ax=ax)
        contours.append(cont)
    
    # Make figure layout tight
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    def update(frame):
        """Update function for animation"""
        # Update simulation
        self._update()
        
        # Update each contour plot
        for ax, cont, (title, (field, cmap)) in zip(axes, contours, cell_types.items()):
            # Clear previous contours
            for coll in cont.collections:
                coll.remove()
                
            # Create new contours
            slice_data = field[slice_pos, :, :]
            new_cont = ax.contourf(
                slice_data,
                levels=np.linspace(0, max(0.0001, slice_data.max()), 20),
                cmap=cmap
            )
            contours[axes.tolist().index(ax)] = new_cont
            
            # Update title with max value
            ax.set_title(f'{title}\nMax: {slice_data.max():.4f}')
        
        return contours
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=interval,
        blit=False,
        repeat=False
    )
    
    plt.show()
    
    return anim  # Return animation object to prevent garbage collection

def animate_single_slice(self, steps=100, slice_pos=None, interval=100):
    """
    Create an animation of a single contour plot showing all cell types overlaid.
    
    Parameters:
    -----------
    steps : int
        Number of frames in the animation
    slice_pos : int, optional
        Position of x-slice. If None, uses middle of grid
    interval : int
        Delay between frames in milliseconds
    """
    import matplotlib.animation as animation
    
    # Set default slice position to middle of grid if not specified
    if slice_pos is None:
        slice_pos = self.grid_shape[0] // 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(f'Combined Tumor Growth Animation (X-Slice at {slice_pos})', fontsize=16)
    
    # Dictionary of cell types and their properties
    cell_types = {
        'Stem Cells': (self.C_S, 'red'),
        'Progenitor Cells': (self.C_P, 'blue'),
        'Differentiated Cells': (self.C_D, 'green'),
        'Necrotic Cells': (self.C_N, 'black')
    }
    
    # Initialize contour plots
    contours = []
    for title, (field, color) in cell_types.items():
        # Create initial contour plot
        cont = ax.contour(
            np.zeros((self.grid_shape[1], self.grid_shape[2])),
            levels=[0.1],  # Single level for clarity
            colors=[color],
            alpha=0.7,
            linewidths=2
        )
        contours.append(cont)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], color=color, label=title)
                      for title, (_, color) in cell_types.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    def update(frame):
        """Update function for animation"""
        # Update simulation
        self._update()
        
        # Clear previous contours
        for cont in contours:
            for coll in cont.collections:
                coll.remove()
        
        # Update each contour plot
        for i, (title, (field, color)) in enumerate(cell_types.items()):
            slice_data = field[slice_pos, :, :]
            if slice_data.max() > 0:  # Only draw contours if cells exist
                new_cont = ax.contour(
                    slice_data,
                    levels=[max(0.1, slice_data.max() * 0.2)],  # Adaptive threshold
                    colors=[color],
                    alpha=0.7,
                    linewidths=2
                )
                contours[i] = new_cont
        
        # Update title with frame number
        ax.set_title(f'Frame {frame}')
        
        return contours
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=interval,
        blit=False,
        repeat=False
    )
    
    plt.show()
    
    return anim  # Return animation object to prevent garbage collection
