import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

class ObservableUtils:
    def __init__(self, grid_size, dx=1.0):
        """
        Initialize the observable utilities.
        
        Args:
            grid_size: Tuple of (nx, ny, nz) grid dimensions
            dx: Grid spacing
        """
        self.grid_size = grid_size
        self.dx = dx
        self.nx, self.ny, self.nz = grid_size
        
        # Create coordinate grids
        self.x = np.arange(self.nx) * self.dx
        self.y = np.arange(self.ny) * self.dx
        self.z = np.arange(self.nz) * self.dx
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
    
    def calculate_radius(self, density_field, threshold=0.1, method='contour'):
        """
        Calculate the effective radius of a tumor from its density field.
        
        Args:
            density_field: 3D density field array
            threshold: Density threshold for boundary detection
            method: Method for radius calculation ('contour', 'mass', 'gyration')
            
        Returns:
            radius: Effective radius
        """
        if method == 'contour':
            return self._calculate_radius_contour(density_field, threshold)
        elif method == 'mass':
            return self._calculate_radius_mass(density_field, threshold)
        elif method == 'gyration':
            return self._calculate_radius_gyration(density_field)
        else:
            raise ValueError(f"Unknown radius calculation method: {method}")
    
    def _calculate_radius_contour(self, density_field, threshold):
        """
        Calculate radius using contour method (distance from center to boundary).
        """
        # Find center of mass
        com = self.calculate_center_of_mass(density_field)
        
        # Create binary mask above threshold
        mask = density_field > threshold
        
        if not np.any(mask):
            return 0.0
        
        # Find boundary points
        boundary_points = self._find_boundary_points(mask)
        
        if len(boundary_points) == 0:
            return 0.0
        
        # Calculate distances from center to boundary points
        center_point = np.array([com[0], com[1], com[2]])
        distances = cdist([center_point], boundary_points)[0]
        
        # Return average radius
        return np.mean(distances)
    
    def _calculate_radius_mass(self, density_field, threshold):
        """
        Calculate radius using mass method (radius of sphere with same volume).
        """
        # Calculate total volume above threshold
        volume = np.sum(density_field > threshold) * (self.dx ** 3)
        
        # Calculate radius of equivalent sphere
        radius = (3 * volume / (4 * np.pi)) ** (1/3)
        
        return radius
    
    def _calculate_radius_gyration(self, density_field):
        """
        Calculate radius of gyration.
        """
        # Find center of mass
        com = self.calculate_center_of_mass(density_field)
        
        # Calculate radius of gyration
        total_mass = np.sum(density_field)
        
        if total_mass == 0:
            return 0.0
        
        # Calculate distances from center of mass
        dx = self.X - com[0]
        dy = self.Y - com[1]
        dz = self.Z - com[2]
        r_squared = dx**2 + dy**2 + dz**2
        
        # Calculate radius of gyration
        radius_gyration = np.sqrt(np.sum(density_field * r_squared) / total_mass)
        
        return radius_gyration
    
    def _find_boundary_points(self, mask):
        """
        Find boundary points of a binary mask.
        """
        # Use morphological operations to find boundary
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded
        
        # Get coordinates of boundary points
        boundary_coords = np.where(boundary)
        if len(boundary_coords[0]) == 0:
            return np.array([])
        
        # Convert to physical coordinates
        boundary_points = np.column_stack([
            boundary_coords[0] * self.dx,
            boundary_coords[1] * self.dx,
            boundary_coords[2] * self.dx
        ])
        
        return boundary_points
    
    def calculate_center_of_mass(self, density_field):
        """
        Calculate the center of mass of a density field.
        
        Args:
            density_field: 3D density field array
            
        Returns:
            com: Tuple of (x, y, z) coordinates of center of mass
        """
        total_mass = np.sum(density_field)
        
        if total_mass == 0:
            # Return center of grid if no mass
            return (self.nx * self.dx / 2, self.ny * self.dx / 2, self.nz * self.dx / 2)
        
        # Calculate weighted average of coordinates
        com_x = np.sum(density_field * self.X) / total_mass
        com_y = np.sum(density_field * self.Y) / total_mass
        com_z = np.sum(density_field * self.Z) / total_mass
        
        return (com_x, com_y, com_z)
    
    def calculate_total_density(self, density_field, normalize_by_volume=False):
        """
        Calculate total density (sum over all voxels).
        
        Args:
            density_field: 3D density field array
            normalize_by_volume: Whether to normalize by total volume
            
        Returns:
            total_density: Total density value
        """
        total_density = np.sum(density_field)
        
        if normalize_by_volume:
            total_volume = self.nx * self.ny * self.nz * (self.dx ** 3)
            total_density /= total_volume
        
        return total_density
    
    def calculate_compactness(self, density_field, threshold=0.1):
        """
        Calculate tumor compactness (surface area to volume ratio).
        
        Args:
            density_field: 3D density field array
            threshold: Density threshold for boundary detection
            
        Returns:
            compactness: Compactness value (lower = more compact)
        """
        # Create binary mask above threshold
        mask = density_field > threshold
        
        if not np.any(mask):
            return 0.0
        
        # Calculate volume
        volume = np.sum(mask) * (self.dx ** 3)
        
        # Calculate surface area using morphological operations
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded
        surface_area = np.sum(boundary) * (self.dx ** 2)
        
        if volume == 0:
            return 0.0
        
        # Calculate compactness (surface area to volume ratio)
        compactness = surface_area / volume
        
        return compactness
    
    def calculate_eccentricity(self, density_field, threshold=0.1):
        """
        Calculate tumor eccentricity (deviation from spherical shape).
        
        Args:
            density_field: 3D density field array
            threshold: Density threshold for boundary detection
            
        Returns:
            eccentricity: Eccentricity value (0 = spherical, 1 = highly elongated)
        """
        # Find center of mass
        com = self.calculate_center_of_mass(density_field)
        
        # Create binary mask above threshold
        mask = density_field > threshold
        
        if not np.any(mask):
            return 0.0
        
        # Calculate moments of inertia
        dx = self.X - com[0]
        dy = self.Y - com[1]
        dz = self.Z - com[2]
        
        # Calculate second moments
        Ixx = np.sum(mask * (dy**2 + dz**2))
        Iyy = np.sum(mask * (dx**2 + dz**2))
        Izz = np.sum(mask * (dx**2 + dy**2))
        
        # Calculate principal moments
        moments = np.array([Ixx, Iyy, Izz])
        moments = np.sort(moments)[::-1]  # Sort in descending order
        
        if moments[0] == 0:
            return 0.0
        
        # Calculate eccentricity
        eccentricity = np.sqrt(1 - (moments[2] / moments[0]))
        
        return eccentricity
    
    def calculate_surface_area(self, density_field, threshold=0.1):
        """
        Calculate tumor surface area.
        
        Args:
            density_field: 3D density field array
            threshold: Density threshold for boundary detection
            
        Returns:
            surface_area: Surface area
        """
        # Create binary mask above threshold
        mask = density_field > threshold
        
        if not np.any(mask):
            return 0.0
        
        # Calculate surface area using morphological operations
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded
        surface_area = np.sum(boundary) * (self.dx ** 2)
        
        return surface_area
    
    def calculate_volume(self, density_field, threshold=0.1):
        """
        Calculate tumor volume.
        
        Args:
            density_field: 3D density field array
            threshold: Density threshold for boundary detection
            
        Returns:
            volume: Volume
        """
        # Create binary mask above threshold
        mask = density_field > threshold
        
        volume = np.sum(mask) * (self.dx ** 3)
        
        return volume
    
    def calculate_sphericity(self, density_field, threshold=0.1):
        """
        Calculate tumor sphericity (how close to spherical shape).
        
        Args:
            density_field: 3D density field array
            threshold: Density threshold for boundary detection
            
        Returns:
            sphericity: Sphericity value (1 = perfect sphere, 0 = highly irregular)
        """
        volume = self.calculate_volume(density_field, threshold)
        surface_area = self.calculate_surface_area(density_field, threshold)
        
        if volume == 0 or surface_area == 0:
            return 0.0
        
        # Calculate sphericity
        sphericity = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / surface_area
        
        return sphericity
    
    def calculate_fractal_dimension(self, density_field, threshold=0.1, box_sizes=None):
        """
        Calculate fractal dimension using box-counting method.
        
        Args:
            density_field: 3D density field array
            threshold: Density threshold for boundary detection
            box_sizes: List of box sizes to use (default: powers of 2)
            
        Returns:
            fractal_dimension: Fractal dimension
        """
        # Create binary mask above threshold
        mask = density_field > threshold
        
        if not np.any(mask):
            return 0.0
        
        # Default box sizes
        if box_sizes is None:
            max_size = min(self.nx, self.ny, self.nz)
            box_sizes = [2**i for i in range(1, int(np.log2(max_size)) + 1)]
        
        # Count boxes for each size
        box_counts = []
        for size in box_sizes:
            count = self._box_count(mask, size)
            box_counts.append(count)
        
        # Calculate fractal dimension using linear regression
        if len(box_counts) < 2:
            return 0.0
        
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        
        # Linear regression
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        fractal_dimension = -slope
        
        return fractal_dimension
    
    def _box_count(self, mask, box_size):
        """
        Count number of boxes of given size that contain any part of the mask.
        """
        nx, ny, nz = mask.shape
        count = 0
        
        for i in range(0, nx, box_size):
            for j in range(0, ny, box_size):
                for k in range(0, nz, box_size):
                    # Extract box
                    box = mask[i:min(i+box_size, nx), 
                              j:min(j+box_size, ny), 
                              k:min(k+box_size, nz)]
                    
                    # Check if box contains any part of the mask
                    if np.any(box):
                        count += 1
        
        return count
    
    def calculate_growth_rate(self, density_field_1, density_field_2, time_1, time_2, 
                            threshold=0.1, method='radius'):
        """
        Calculate growth rate between two time points.
        
        Args:
            density_field_1: Density field at time 1
            density_field_2: Density field at time 2
            time_1: Time at first measurement
            time_2: Time at second measurement
            threshold: Density threshold for calculations
            method: Method for growth rate calculation ('radius', 'volume', 'density')
            
        Returns:
            growth_rate: Growth rate per unit time
        """
        dt = time_2 - time_1
        
        if dt <= 0:
            return 0.0
        
        if method == 'radius':
            radius_1 = self.calculate_radius(density_field_1, threshold)
            radius_2 = self.calculate_radius(density_field_2, threshold)
            growth_rate = (radius_2 - radius_1) / dt
        elif method == 'volume':
            volume_1 = self.calculate_volume(density_field_1, threshold)
            volume_2 = self.calculate_volume(density_field_2, threshold)
            growth_rate = (volume_2 - volume_1) / dt
        elif method == 'density':
            density_1 = self.calculate_total_density(density_field_1)
            density_2 = self.calculate_total_density(density_field_2)
            growth_rate = (density_2 - density_1) / dt
        else:
            raise ValueError(f"Unknown growth rate method: {method}")
        
        return growth_rate
    
    def calculate_all_observables(self, density_field, threshold=0.1):
        """
        Calculate all observables for a given density field.
        
        Args:
            density_field: 3D density field array
            threshold: Density threshold for calculations
            
        Returns:
            observables: Dictionary containing all calculated observables
        """
        observables = {}
        
        # Basic observables
        observables['total_density'] = self.calculate_total_density(density_field)
        observables['volume'] = self.calculate_volume(density_field, threshold)
        observables['surface_area'] = self.calculate_surface_area(density_field, threshold)
        
        # Geometric observables
        observables['radius_contour'] = self.calculate_radius(density_field, threshold, 'contour')
        observables['radius_mass'] = self.calculate_radius(density_field, threshold, 'mass')
        observables['radius_gyration'] = self.calculate_radius(density_field, threshold, 'gyration')
        
        # Shape observables
        observables['center_of_mass'] = self.calculate_center_of_mass(density_field)
        observables['compactness'] = self.calculate_compactness(density_field, threshold)
        observables['eccentricity'] = self.calculate_eccentricity(density_field, threshold)
        observables['sphericity'] = self.calculate_sphericity(density_field, threshold)
        
        # Advanced observables
        observables['fractal_dimension'] = self.calculate_fractal_dimension(density_field, threshold)
        
        return observables
