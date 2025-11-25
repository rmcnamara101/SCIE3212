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
    
    def calculate_radius(self, density_field, threshold=0.1, method='contour', percentile=50.0):
        """
        Calculate the effective radius of a tumor from its density field.
        
        Args:
            density_field: 3D density field array
            threshold: Density threshold for boundary detection
            method: Method for radius calculation ('contour', 'mass', 'gyration')
            percentile: For 'contour' method, percentile of boundary distances to use (50.0 = median/mean-like, 
                       95.0 = 95th percentile to ignore outliers, 100.0 = maximum)
            
        Returns:
            radius: Effective radius
        """
        if method == 'contour':
            return self._calculate_radius_contour(density_field, threshold, percentile)
        elif method == 'mass':
            return self._calculate_radius_mass(density_field, threshold)
        elif method == 'gyration':
            return self._calculate_radius_gyration(density_field)
        else:
            raise ValueError(f"Unknown radius calculation method: {method}")
    
    def _calculate_radius_contour(self, density_field, threshold, percentile=50.0):
        """
        Calculate radius using contour method (distance from center to boundary).
        
        Uses percentile of boundary distances to reduce sensitivity to scattered outliers.
        Default percentile=50.0 gives mean-like behavior, higher percentiles (e.g., 95.0) 
        ignore far outliers, 100.0 gives maximum radius.
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
        
        # Return percentile of distances (50.0 ≈ mean for symmetric distributions, 
        # higher percentiles ignore outliers from scattered boundary points)
        if percentile == 50.0:
            # Use mean for backward compatibility
            return np.mean(distances)
        elif percentile == 100.0:
            # Maximum radius
            return np.max(distances)
        else:
            # Use specified percentile
            return np.percentile(distances, percentile)
    
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
        # Filter out NaN and Inf values
        valid_mask = np.isfinite(density_field)
        if not np.any(valid_mask):
            # Return center of grid if no valid mass
            return (self.nx * self.dx / 2, self.ny * self.dx / 2, self.nz * self.dx / 2)
        
        # Use only valid values for calculation
        valid_density = np.where(valid_mask, density_field, 0.0)
        total_mass = np.sum(valid_density)
        
        if total_mass == 0 or not np.isfinite(total_mass):
            # Return center of grid if no mass or invalid mass
            return (self.nx * self.dx / 2, self.ny * self.dx / 2, self.nz * self.dx / 2)
        
        # Calculate weighted average of coordinates
        com_x = np.sum(valid_density * self.X) / total_mass
        com_y = np.sum(valid_density * self.Y) / total_mass
        com_z = np.sum(valid_density * self.Z) / total_mass
        
        # Check if results are valid
        if not (np.isfinite(com_x) and np.isfinite(com_y) and np.isfinite(com_z)):
            # Return center of grid if calculation failed
            return (self.nx * self.dx / 2, self.ny * self.dx / 2, self.nz * self.dx / 2)
        
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
    
    def _calculate_adaptive_threshold(self, source_terms, density_field, 
                                     threshold_type='percentile', threshold_value=75.0,
                                     density_threshold=0.1):
        """
        Calculate adaptive threshold based on source term distribution.
        
        Args:
            source_terms: Source term field (nx, ny, nz)
            density_field: Cell density field for masking
            threshold_type: Type of adaptive threshold ('percentile', 'relative_max', 'zero_crossing')
            threshold_value: Value for threshold calculation (percentile or fraction)
            density_threshold: Minimum density to consider as tumor region
            
        Returns:
            adaptive_threshold: Calculated adaptive threshold value
        """
        # Mask to tumor region only
        tumor_mask = density_field > density_threshold
        
        if not np.any(tumor_mask):
            return 0.0
        
        # Get source terms in tumor region
        tumor_source_terms = source_terms[tumor_mask]
        
        if len(tumor_source_terms) == 0:
            return 0.0
        
        if threshold_type == 'percentile':
            # Use percentile of source terms in tumor region
            adaptive_threshold = np.percentile(tumor_source_terms, threshold_value)
        elif threshold_type == 'relative_max':
            # Use fraction of maximum source term in tumor region
            max_source = np.max(tumor_source_terms)
            adaptive_threshold = threshold_value * max_source
        elif threshold_type == 'relative_mean':
            # Use fraction of mean source term in tumor region
            mean_source = np.mean(tumor_source_terms)
            adaptive_threshold = threshold_value * mean_source
        elif threshold_type == 'zero_crossing':
            # For necrotic zone: use zero or slightly above zero
            adaptive_threshold = 0.0 if threshold_value == 0 else threshold_value
        elif threshold_type == 'absolute':
            # Use absolute value directly (for backward compatibility)
            adaptive_threshold = threshold_value
        else:
            raise ValueError(f"Unknown threshold_type: {threshold_type}")
        
        return adaptive_threshold
    
    def calculate_inhibited_radius(self, source_terms, density_field, 
                                   phi_hat=None,
                                   density_threshold=0.2, 
                                   necrotic_threshold=0.01,
                                   method='radial_average',
                                   # Deprecated parameters kept for backward compatibility
                                   nutrient_field=None,
                                   lambda_rates=None,
                                   nutrient_thresholds=None,
                                   k_switch=10.0,
                                   beta_N=0.0005,
                                   proliferation_threshold=0.01,
                                   death_switch_threshold=0.7,
                                   growth_threshold=2.0,
                                   threshold_type='percentile',
                                   threshold_percentile=75.0,
                                   use_adaptive_threshold=True,
                                   exclude_necrotic=True,
                                   drop_fraction=0.5):
        """
        Calculate the inhibited radius - the region containing inhibited or dead cells.
        
        Based on paper definition: "Proportion of outer radius containing inhibited or dead cells."
        This uses the necrotic population (phi_hat[-1]) to find where necrotic cells exist.
        
        Args:
            source_terms: Source term field (M, nx, ny, nz) - kept for backward compatibility
            density_field: Cell density field for tumor center detection
            phi_hat: Cell fraction fields for all populations (M, nx, ny, nz) - required
            density_threshold: Minimum density to consider as tumor region (default 0.1)
            necrotic_threshold: Threshold for necrotic population density (default 0.01)
            method: Method for calculation ('radial_average', 'contour')
            [Deprecated parameters - kept for backward compatibility]
            
        Returns:
            inhibited_radius: Outermost radius where necrotic population > threshold
        """
        # Check if phi_hat is provided
        if phi_hat is None:
            # Fall back to old behavior for backward compatibility
            return self._calculate_inhibited_radius_old(
                source_terms, density_field, density_threshold, method
            )
        
        # Validate phi_hat shape
        if phi_hat.ndim != 4:
            raise ValueError(f"phi_hat must be 4D (M, nx, ny, nz), got shape {phi_hat.shape}")
        
        if phi_hat.shape[0] < 2:
            # Need at least 2 populations (one viable + necrotic)
            return 0.0
        
        # Get necrotic population (last population)
        necrotic_density = phi_hat[-1]
        
        # Find tumor center using density field
        com = self.calculate_center_of_mass(density_field)
        center_point = np.array([com[0], com[1], com[2]])
        
        # Calculate total tumor radius to ensure inhibited radius is within tumor boundary
        radius_method = method if method in ['contour', 'mass', 'gyration'] else 'contour'
        total_radius = self.calculate_radius(density_field, threshold=density_threshold, method=radius_method)
        
        # Check if there is any tumor density
        tumor_mask = density_field > density_threshold
        if not np.any(tumor_mask):
            return 0.0
        
        # Inhibited region: where necrotic population > threshold (contains inhibited/dead cells)
        inhibited_mask = (necrotic_density > necrotic_threshold) & tumor_mask
        
        if not np.any(inhibited_mask):
            # No inhibited region found
            return 0.0
        
        # Calculate radial distances from center
        r_squared = (self.X - com[0])**2 + (self.Y - com[1])**2 + (self.Z - com[2])**2
        radial_distances = np.sqrt(r_squared)
        
        if method == 'radial_average':
            # Bin data by radial distance and find outer boundary of inhibited region
            max_radius = np.max(radial_distances)
            num_bins = max(10, min(200, int(max_radius / self.dx)))
            radial_bins = np.linspace(0, max_radius, num_bins + 1)
            
            # Calculate average necrotic density and total density in each radial bin
            avg_necrotic_density = np.zeros(num_bins)
            avg_density = np.zeros(num_bins)
            
            for i in range(num_bins):
                r_min = radial_bins[i]
                r_max = radial_bins[i + 1]
                mask = (radial_distances >= r_min) & (radial_distances < r_max)
                
                if np.any(mask):
                    avg_necrotic_density[i] = np.mean(necrotic_density[mask])
                    avg_density[i] = np.mean(density_field[mask])
            
            # Work only where tumor density is present
            viable_mask = avg_density > density_threshold
            if not np.any(viable_mask):
                return 0.0
            
            radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
            
            # Find outer boundary of inhibited region (moving outward from center)
            # Inhibited region: necrotic density > threshold
            inhibited_radius = 0.0
            in_inhibited = False
            
            for i in range(num_bins):
                if not viable_mask[i]:
                    continue
                
                is_inhibited = avg_necrotic_density[i] > necrotic_threshold
                
                if is_inhibited:
                    in_inhibited = True
                    inhibited_radius = radial_centers[i]
                elif in_inhibited and not is_inhibited:
                    # We've exited the inhibited region, keep the last inhibited radius
                    break
            
            # Ensure inhibited radius is within tumor boundary
            if inhibited_radius > total_radius:
                inhibited_radius = 0.0
                
        elif method == 'contour':
            # Use contour method: find maximum distance to points in inhibited region
            inhibited_coords = np.where(inhibited_mask)
            if len(inhibited_coords[0]) == 0:
                return 0.0
            
            inhibited_points = np.column_stack([
                inhibited_coords[0] * self.dx,
                inhibited_coords[1] * self.dx,
                inhibited_coords[2] * self.dx
            ])
            
            # Calculate distances from center to all inhibited points
            distances = cdist([center_point], inhibited_points)[0]
            
            # The inhibited radius is the maximum distance to inhibited points (outer boundary)
            inhibited_radius = np.max(distances) if len(distances) > 0 else 0.0
            
            # Ensure inhibited radius is within tumor boundary
            if inhibited_radius > total_radius:
                inhibited_radius = 0.0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return inhibited_radius

    def _calculate_inhibited_radius_old(self, source_terms, density_field, density_threshold, method):
        """
        Old implementation for backward compatibility (uses necrotic source terms).
        """
        # Extract necrotic source terms (last population)
        if source_terms.ndim == 4:
            if source_terms.shape[0] == 0:
                return 0.0
            necrotic_source_terms = source_terms[-1]
        else:
            necrotic_source_terms = source_terms
        
        # Find tumor center
        com = self.calculate_center_of_mass(density_field)
        center_point = np.array([com[0], com[1], com[2]])
        
        # Calculate total tumor radius
        radius_method = method if method in ['contour', 'mass', 'gyration'] else 'contour'
        total_radius = self.calculate_radius(density_field, threshold=density_threshold, method=radius_method)
        
        # Check if there are any positive necrotic source terms
        tumor_mask = density_field > density_threshold
        if not np.any(tumor_mask):
            return 0.0
        
        max_necrotic_source = np.max(necrotic_source_terms[tumor_mask])
        if max_necrotic_source <= 0:
            return 0.0
        
        zero_tolerance = max(1e-6, 0.01 * max_necrotic_source)
        r_squared = (self.X - com[0])**2 + (self.Y - com[1])**2 + (self.Z - com[2])**2
        radial_distances = np.sqrt(r_squared)
        
        if method == 'radial_average':
            max_radius = np.max(radial_distances)
            num_bins = max(10, min(200, int(max_radius / self.dx)))
            radial_bins = np.linspace(0, max_radius, num_bins + 1)
            
            avg_necrotic_sources = np.zeros(num_bins)
            avg_density = np.zeros(num_bins)
            
            for i in range(num_bins):
                r_min = radial_bins[i]
                r_max = radial_bins[i + 1]
                mask = (radial_distances >= r_min) & (radial_distances < r_max)
                
                if np.any(mask):
                    avg_necrotic_sources[i] = np.mean(necrotic_source_terms[mask])
                    avg_density[i] = np.mean(density_field[mask])
            
            viable_mask = avg_density > density_threshold
            if not np.any(viable_mask):
                return 0.0
            
            radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
            
            smoothed = avg_necrotic_sources.copy()
            for i in range(1, num_bins - 1):
                if viable_mask[i - 1] or viable_mask[i] or viable_mask[i + 1]:
                    smoothed[i] = (avg_necrotic_sources[i - 1] + avg_necrotic_sources[i] + avg_necrotic_sources[i + 1]) / 3.0
            
            inhibited_radius = 0.0
            found = False
            for i in range(1, num_bins):
                if not (viable_mask[i - 1] and viable_mask[i]):
                    continue
                if (np.abs(smoothed[i - 1]) <= zero_tolerance or smoothed[i - 1] < 0) and (smoothed[i] > zero_tolerance) and (radial_centers[i] <= total_radius):
                    inhibited_radius = radial_centers[i]
                    found = True
                    break
            
            if not found:
                positive_mask = (avg_necrotic_sources > zero_tolerance) & viable_mask
                if np.any(positive_mask):
                    inhibited_radius = 0.0
                else:
                    inhibited_radius = 0.0
            
            if inhibited_radius > total_radius:
                inhibited_radius = 0.0
                
        elif method == 'contour':
            positive_mask = (necrotic_source_terms > zero_tolerance) & (density_field > density_threshold)
            
            if not np.any(positive_mask):
                return 0.0
            
            positive_coords = np.where(positive_mask)
            positive_points = np.column_stack([
                positive_coords[0] * self.dx,
                positive_coords[1] * self.dx,
                positive_coords[2] * self.dx
            ])
            
            distances = cdist([center_point], positive_points)[0]
            inhibited_radius = np.min(distances) if len(distances) > 0 else 0.0
            
            if inhibited_radius > total_radius:
                inhibited_radius = 0.0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return inhibited_radius
    
    def calculate_necrotic_radius(self, source_terms, density_field,
                                  phi_hat=None,
                                  density_threshold=0.1,
                                  viable_threshold=0.01,
                                  method='radial_average',
                                  # Deprecated parameters kept for backward compatibility
                                  necrotic_threshold=0.0,
                                  threshold_type='zero_crossing',
                                  threshold_percentile=25.0,
                                  use_adaptive_threshold=True,
                                  exclude_necrotic=True):
        """
        Calculate the necrotic radius - the boundary of the necrotic core.
        
        The necrotic core is defined as the region where necrotic cells dominate
        (necrotic fraction >= 0.9, meaning essentially free of viable cells).
        This finds the outermost radius where necrotic cells make up at least 90% of the cell population.
        
        Args:
            source_terms: Source term field (M, nx, ny, nz) - kept for backward compatibility
            density_field: Cell density field for tumor center detection
            phi_hat: Cell fraction fields for all populations (M, nx, ny, nz) - required
            density_threshold: Minimum density to consider as tumor region (default 0.1)
            viable_threshold: Threshold for viable cell density (default 0.01) - kept for backward compatibility
            method: Method for calculation ('radial_average', 'contour')
            [Deprecated parameters - kept for backward compatibility]
            
        Returns:
            necrotic_radius: Outermost radius where necrotic fraction >= 0.9 (necrotic cells dominate)
        """
        # Check if phi_hat is provided
        if phi_hat is None:
            # Fall back to old behavior for backward compatibility
            # Use source terms method
            return self._calculate_necrotic_radius_from_source_terms(
                source_terms, density_field, necrotic_threshold, density_threshold,
                method, threshold_type, threshold_percentile, use_adaptive_threshold, exclude_necrotic
            )
        
        # Validate phi_hat shape
        if phi_hat.ndim != 4:
            raise ValueError(f"phi_hat must be 4D (M, nx, ny, nz), got shape {phi_hat.shape}")
        
        if phi_hat.shape[0] < 2:
            # Need at least 2 populations (one viable + necrotic)
            return 0.0
        
        # Calculate viable cell density (all populations except necrotic)
        viable_density = np.sum(phi_hat[:-1], axis=0)
        
        # Also get necrotic population density (last population) for alternative calculation
        necrotic_density = phi_hat[-1] if phi_hat.shape[0] > 1 else np.zeros_like(viable_density)
        
        # Find tumor center using density field
        com = self.calculate_center_of_mass(density_field)
        center_point = np.array([com[0], com[1], com[2]])
        
        # Calculate radial distances from center
        r_squared = (self.X - com[0])**2 + (self.Y - com[1])**2 + (self.Z - com[2])**2
        radial_distances = np.sqrt(r_squared)
        
        # Only consider points within tumor (where total density > threshold)
        # Also filter out NaN/Inf values
        tumor_mask = (density_field > density_threshold) & np.isfinite(density_field) & np.isfinite(radial_distances)
        if not np.any(tumor_mask):
            return 0.0
        
        if method == 'radial_average':
            # Bin data by radial distance and find transition from necrotic core to viable rim
            valid_radii = radial_distances[tumor_mask]
            if len(valid_radii) == 0 or not np.all(np.isfinite(valid_radii)):
                return 0.0
            
            max_radius = np.max(valid_radii)
            if not np.isfinite(max_radius) or max_radius <= 0:
                return 0.0
            
            # Additional safety check for dx
            if not np.isfinite(self.dx) or self.dx <= 0:
                # Fallback to reasonable default
                dx_safe = 1.0
            else:
                dx_safe = self.dx
            
            # Calculate number of bins with additional safety
            try:
                bins_float = max_radius / dx_safe
                if not np.isfinite(bins_float):
                    return 0.0
                num_bins = max(10, min(200, int(bins_float)))
            except (ValueError, OverflowError):
                return 0.0
            radial_bins = np.linspace(0, max_radius, num_bins + 1)
            radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
            
            # Calculate average viable density, necrotic density, and total density in each radial bin
            avg_viable_density = np.zeros(num_bins)
            avg_necrotic_density = np.zeros(num_bins)
            avg_total_density = np.zeros(num_bins)
            
            for i in range(num_bins):
                r_min = radial_bins[i]
                r_max = radial_bins[i + 1]
                mask = (radial_distances >= r_min) & (radial_distances < r_max) & tumor_mask
                
                if np.any(mask):
                    avg_viable_density[i] = np.mean(viable_density[mask])
                    avg_necrotic_density[i] = np.mean(necrotic_density[mask])
                    avg_total_density[i] = np.mean(density_field[mask])
            
            # Find the outermost radius of the necrotic core
            # The necrotic core is defined as the region where necrotic cells dominate
            # (necrotic fraction ≈ 0.9-1.0, meaning essentially free of viable cells)
            # We want the outermost radius where necrotic_fraction >= threshold (default 0.9)
            necrotic_radius = 0.0
            
            # Threshold for necrotic fraction: core is where necrotic cells make up at least this fraction
            # Default 0.9 means 90% necrotic cells (essentially free of viable cells)
            necrotic_fraction_threshold = 0.9
            
            for i in range(num_bins):
                # Only consider bins with significant tumor density
                if avg_total_density[i] < density_threshold * 0.5:
                    # If we've found a core and hit low density, we've reached the edge
                    if necrotic_radius > 0:
                        break
                    continue
                
                # Calculate the fraction of necrotic cells in this radial bin
                # necrotic_fraction = necrotic_density / total_density
                necrotic_fraction = avg_necrotic_density[i] / max(avg_total_density[i], 1e-6)
                
                # Check if we're in the necrotic core (necrotic cells dominate)
                if necrotic_fraction >= necrotic_fraction_threshold:
                    # Update to the outermost radius where necrotic cells dominate
                    necrotic_radius = radial_centers[i]
                elif necrotic_radius > 0:
                    # We've exited the necrotic core, keep the last necrotic radius
                    break
            
        elif method == 'contour':
            # Use contour method: find outermost boundary of necrotic core
            # Necrotic core: where necrotic fraction >= 0.9 (necrotic cells dominate)
            necrotic_fraction = necrotic_density / np.maximum(density_field, 1e-6)
            necrotic_mask = (necrotic_fraction >= 0.9) & tumor_mask
            
            if not np.any(necrotic_mask):
                return 0.0
            
            # Find all points in the necrotic core
            necrotic_coords = np.where(necrotic_mask)
            necrotic_points = np.column_stack([
                necrotic_coords[0] * self.dx,
                necrotic_coords[1] * self.dx,
                necrotic_coords[2] * self.dx
            ])
            
            # Calculate distances from center to all necrotic core points
            distances = cdist([center_point], necrotic_points)[0]
            
            # The necrotic radius is the maximum distance to necrotic core points
            necrotic_radius = np.max(distances) if len(distances) > 0 else 0.0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return necrotic_radius
    
    def _calculate_necrotic_radius_from_source_terms(self, source_terms, density_field,
                                                     necrotic_threshold, density_threshold,
                                                     method, threshold_type, threshold_percentile,
                                                     use_adaptive_threshold, exclude_necrotic):
        """
        Old implementation using source terms (for backward compatibility).
        """
        # Aggregate source terms if multi-population
        if source_terms.ndim == 4:
            if exclude_necrotic and source_terms.shape[0] > 1:
                total_source_terms = np.sum(source_terms[:-1], axis=0)
            else:
                total_source_terms = np.sum(source_terms, axis=0)
        else:
            total_source_terms = source_terms
        
        # Calculate adaptive threshold if requested
        if use_adaptive_threshold:
            if threshold_type == 'zero_crossing':
                necrotic_threshold = 0.0
            else:
                necrotic_threshold = self._calculate_adaptive_threshold(
                    total_source_terms, density_field,
                    threshold_type=threshold_type,
                    threshold_value=threshold_percentile,
                    density_threshold=density_threshold
                )
        
        # Find tumor center using density field
        com = self.calculate_center_of_mass(density_field)
        center_point = np.array([com[0], com[1], com[2]])
        
        # Calculate radial distances from center
        r_squared = (self.X - com[0])**2 + (self.Y - com[1])**2 + (self.Z - com[2])**2
        radial_distances = np.sqrt(r_squared)
        
        if method == 'radial_average':
            max_radius = np.max(radial_distances)
            num_bins = max(10, min(200, int(max_radius / self.dx)))
            radial_bins = np.linspace(0, max_radius, num_bins + 1)
            
            avg_source_terms = np.zeros(num_bins)
            avg_density = np.zeros(num_bins)
            
            for i in range(num_bins):
                r_min = radial_bins[i]
                r_max = radial_bins[i + 1]
                mask = (radial_distances >= r_min) & (radial_distances < r_max)
                
                if np.any(mask):
                    avg_source_terms[i] = np.mean(total_source_terms[mask])
                    avg_density[i] = np.mean(density_field[mask])
            
            viable_mask = avg_density > density_threshold
            if not np.any(viable_mask):
                return 0.0
            
            radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
            
            necrotic_radius = 0.0
            found_necrotic = False
            
            necrotic_mask = (avg_source_terms <= necrotic_threshold) & viable_mask
            if np.any(necrotic_mask):
                necrotic_indices = np.where(necrotic_mask)[0]
                if len(necrotic_indices) > 0:
                    max_necrotic_idx = np.max(necrotic_indices)
                    necrotic_radius = radial_centers[max_necrotic_idx]
                    found_necrotic = True
            
            if not found_necrotic:
                necrotic_radius = 0.0
                
        elif method == 'contour':
            necrotic_mask = (total_source_terms <= necrotic_threshold) & (density_field > density_threshold)
            
            if not np.any(necrotic_mask):
                return 0.0
            
            necrotic_coords = np.where(necrotic_mask)
            necrotic_points = np.column_stack([
                necrotic_coords[0] * self.dx,
                necrotic_coords[1] * self.dx,
                necrotic_coords[2] * self.dx
            ])
            
            distances = cdist([center_point], necrotic_points)[0]
            necrotic_radius = np.max(distances) if len(distances) > 0 else 0.0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return necrotic_radius
    
    def calculate_all_observables(self, density_field, threshold=0.1, source_terms=None, 
                                 growth_threshold=0.01, use_adaptive_threshold=True):
        """
        Calculate all observables for a given density field.
        
        Args:
            density_field: 3D density field array
            threshold: Density threshold for calculations
            source_terms: Optional source term field (M, nx, ny, nz) or (nx, ny, nz)
            growth_threshold: Threshold for detecting proliferative cells (for inhibited radius)
            
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
        
        # Growth-related observables (requires source terms)
        if source_terms is not None:
            observables['inhibited_radius'] = self.calculate_inhibited_radius(
                source_terms, density_field, growth_threshold=growth_threshold, 
                density_threshold=threshold, method='radial_average',
                use_adaptive_threshold=use_adaptive_threshold
            )
            observables['necrotic_radius'] = self.calculate_necrotic_radius(
                source_terms, density_field, necrotic_threshold=0.0,
                density_threshold=threshold, method='radial_average',
                use_adaptive_threshold=use_adaptive_threshold
            )
        
        return observables
