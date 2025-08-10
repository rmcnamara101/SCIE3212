# Initial Conditions Configuration

This document describes the new initial conditions system that allows you to specify different types of initial conditions for your tumor growth simulations through YAML configuration files.

## Overview

The initial conditions system has been separated from the main simulation code and is now fully configurable through YAML templates. This allows for easy experimentation with different initial setups without modifying code.

## Configuration Structure

Add an `initial_conditions` section to your YAML configuration file:

```yaml
initial_conditions:
  type: <type_of_initial_conditions>
  # ... type-specific parameters
  seeding_densities:
    <population_label>: <density>
  nutrient:
    type: <nutrient_type>
    # ... nutrient-specific parameters
```

## Supported Initial Condition Types

### 1. Spherical (`"spherical"`)

Creates a spherical tumor in the center of the domain.

**Parameters:**
- `center`: [x, y, z] coordinates (optional, defaults to domain center)
- `radius`: radius of the sphere (optional, defaults to domain_size/4)

**Example:**
```yaml
initial_conditions:
  type: "spherical"
  center: [32, 32, 32]
  radius: 16
  seeding_densities:
    tumor: 0.8
    stroma: 0.2
```

### 2. Deformed Blob (`"gaussian_random_blob"`)

Creates a deformed blob with hard boundaries, simulating realistic cell cluster shapes.

**Parameters:**
- `center`: [x, y, z] coordinates (optional, defaults to domain center)
- `radius`: base radius of the blob (optional, defaults to domain_size/4)
- `deformation_strength`: how much to deform the shape (optional, defaults to 0.3)
  - 0.0 = perfect sphere
  - 0.3 = moderately deformed
  - 0.5 = very deformed

**How it works:**
1. Starts with a base spherical shape
2. Applies spatially correlated deformations to create irregular, blobby boundaries
3. Creates hard boundaries (full density inside, zero outside)
4. Distributes the resulting density among populations

**Example:**
```yaml
initial_conditions:
  type: "gaussian_random_blob"
  center: [32, 32, 32]
  radius: 16  # Base radius of the blob
  deformation_strength: 0.4  # Moderate deformation for realistic shape
  seeding_densities:
    tumor: 0.7
    stroma: 0.3
```

### 3. Multiple Spheres (`"multiple_spheres"`)

Creates multiple spherical regions for different populations.

**Parameters:**
- `spheres`: list of sphere configurations
  - `center`: [x, y, z] coordinates
  - `radius`: radius of the sphere
  - `population`: population index (0-based)
  - `density`: initial density for this sphere

**Example:**
```yaml
initial_conditions:
  type: "multiple_spheres"
  spheres:
    - center: [16, 32, 32]
      radius: 12
      population: 0
      density: 0.8
    - center: [48, 32, 32]
      radius: 10
      population: 1
      density: 0.6
```

### 4. Uniform (`"uniform"`)

Creates uniform initial conditions across the entire domain.

**Parameters:**
- `seeding_densities`: densities for each population

**Example:**
```yaml
initial_conditions:
  type: "uniform"
  seeding_densities:
    tumor: 0.3
    stroma: 0.2
```

### 5. Custom (`"custom"`)

Placeholder for custom initial conditions (currently falls back to spherical).

## Seeding Densities

The `seeding_densities` parameter allows you to specify the initial density for each population:

```yaml
seeding_densities:
  tumor: 0.8      # 80% tumor cells
  stroma: 0.2     # 20% stroma cells
```

If not specified, the system will use default values:
- First population: 0.8
- Other populations: 0.1 / (number_of_populations - 1)

## Nutrient Field Configuration

You can also configure the initial nutrient field:

### Uniform Nutrient (`"uniform"`)
```yaml
nutrient:
  type: "uniform"
  concentration: 1.0
```

### Gradient Nutrient (`"gradient"`)
```yaml
nutrient:
  type: "gradient"
  concentration_max: 1.0
  concentration_min: 0.2
```

If not specified, the system will use the default nutrient manager.

## Example Configurations

See the following example files in the `templates/` directory:
- `example_spherical.yaml` - Spherical tumor example
- `example_gaussian_blob.yaml` - Gaussian random blob example
- `example_multiple_spheres.yaml` - Multiple spheres example

## Testing

Run the test script to see all initial condition types in action:

```bash
python test_initial_conditions.py
```

This will:
1. Load each example configuration
2. Generate the initial conditions
3. Display statistics about the generated fields
4. Create visualization plots saved to the `output/` directory

## Integration with Existing Code

The new system is backward compatible. The `FieldManager.initialize_fields()` method now uses the new `InitialConditions` class by default, but still accepts the old `initial_conditions` parameter for backward compatibility.

## Adding New Initial Condition Types

To add a new initial condition type:

1. Add a new method to the `InitialConditions` class:
   ```python
   def _create_new_type_initial_conditions(self) -> np.ndarray:
       # Implementation here
       pass
   ```

2. Add the type to the main initialization method:
   ```python
   elif self.ic_type == "new_type":
       phi_hat = self._create_new_type_initial_conditions()
   ```

3. Update the template YAML file with the new type and its parameters.

## File Structure

```
src/growkit/Fields/InitialConditions/
├── InitialConditions.py          # Main initial conditions class
└── __init__.py

templates/
├── template.yaml                 # Updated template with initial conditions
├── example_spherical.yaml        # Spherical example
├── example_gaussian_blob.yaml    # Gaussian blob example
└── example_multiple_spheres.yaml # Multiple spheres example

test_initial_conditions.py        # Test script
INITIAL_CONDITIONS_README.md      # This documentation
```

## Benefits

1. **Separation of Concerns**: Initial conditions are now separate from the main simulation logic
2. **Configurability**: Easy to experiment with different initial setups through YAML
3. **Extensibility**: Easy to add new initial condition types
4. **Backward Compatibility**: Existing code continues to work
5. **Documentation**: Clear examples and documentation for each type
6. **Testing**: Comprehensive test suite to verify functionality
