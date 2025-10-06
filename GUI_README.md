# Simulation Analysis GUI

A comprehensive graphical user interface for analyzing tumor growth simulation results. This GUI provides easy access to all plotting capabilities without needing to switch between tabs or scroll through Jupyter notebooks.

## Features

### 🎯 **Easy File Management**
- Load simulation data files (.npz) with a simple file dialog
- Recent files management - quickly reload previously opened simulations
- Automatic file validation and error handling

### 📊 **Comprehensive Plotting**
- **Overview Tab**: Quick access to common plots (radius evolution, density evolution, etc.)
- **Observables Tab**: Detailed analysis of all simulation observables
- **Cell Fields Tab**: Visualize cell density fields for different populations
- **Physics Fields Tab**: Display pressure, velocity, energy, and mass flux fields
- **Nutrient Fields Tab**: Analyze nutrient field evolution and correlations
- **Custom Analysis Tab**: Advanced analysis tools and export options

### ⚙️ **Interactive Controls**
- Real-time parameter adjustment (thresholds, zoom factors, time steps)
- Population selection for multi-population simulations
- Z-slice selection for 3D data visualization
- Plot customization options (colormaps, contours, etc.)

### 💾 **Export Capabilities**
- Save individual plots in multiple formats (PNG, PDF, SVG)
- Export all observables data to CSV
- High-resolution plot output (300 DPI)

## Quick Start

### 1. Launch the GUI
```bash
# From the project root directory
python run_gui.py
```

### 2. Load Simulation Data
- Click "Load Simulation" button
- Navigate to your simulation data file (usually in `laboratory/saved_simulations/`)
- Select a `.npz` file containing your simulation results

### 3. Explore Your Data
- Use the **Overview** tab for quick insights
- Switch to specialized tabs for detailed analysis
- Adjust parameters using the control panels
- Save plots or export data as needed

## Tab Descriptions

### Overview Tab
Quick access to the most commonly used plots:
- **Tumor Radius Evolution**: Track how tumor size changes over time
- **Population Density Evolution**: Monitor cell population changes
- **Tumor Shape Evolution**: Visualize morphological changes
- **Center of Mass Evolution**: Track tumor position changes
- **Compactness Evolution**: Analyze tumor shape complexity
- **All Observables**: Comprehensive overview of all metrics

### Observables Tab
Detailed analysis of simulation observables with customizable parameters.

### Cell Fields Tab
Visualize cell density fields:
- Select specific populations (Stem Cells, Tumor Cells, Necrotic Cells)
- Choose time steps and Z-slices
- Add contours and zoom controls
- Real-time parameter updates

### Physics Fields Tab
Display physical fields from the simulation:
- **Pressure Field**: Hydrostatic pressure distribution
- **Velocity Field**: Solid velocity vectors
- **Energy Derivative**: Adhesion energy gradients
- **Mass Flux**: Cell mass transport
- **Source Terms**: Growth/death rates

### Nutrient Fields Tab
Analyze nutrient dynamics:
- **Field Evolution**: Nutrient concentration over time
- **Statistics Evolution**: Min/max/mean nutrient levels
- **Tumor Correlation**: Relationship between nutrients and tumor density

### Custom Analysis Tab
Advanced analysis tools:
- **Free Energy Evolution**: Track total system energy
- **Population Evolution**: Detailed population dynamics
- **Export Options**: Save data and plots

## Controls and Parameters

### Plot Parameters
- **Threshold**: Density threshold for radius calculations (0.01-0.5)
- **Max Plots**: Maximum number of time points to display (3-12)
- **Zoom Factor**: Zoom level for field visualizations (1.0-5.0)

### Time and Space Controls
- **Time Step**: Select which saved time point to visualize
- **Z-Slice**: Choose which 2D slice to display from 3D data
- **Population**: Select which cell population to analyze

### Display Options
- **Add Contours**: Overlay density contours on field plots
- **Colormaps**: Choose visualization color schemes
- **Save Options**: Export plots in various formats

## File Management

### Recent Files
The GUI automatically tracks recently opened files:
- Use the dropdown to quickly reload previous simulations
- Files are stored in `~/.silicokit_gui_config.json`
- Clear recent files list if needed

### Supported File Formats
- **Input**: `.npz` files containing simulation data
- **Output**: `.png`, `.pdf`, `.svg` for plots; `.csv` for data

## Tips for Best Results

1. **Start with Overview**: Use the Overview tab to get a quick sense of your simulation results
2. **Use Recent Files**: The recent files feature makes it easy to switch between different simulations
3. **Adjust Parameters**: Experiment with different thresholds and zoom levels to find the best visualization
4. **Save Important Plots**: Use the save functionality to preserve key visualizations
5. **Export Data**: Use the export feature to get numerical data for further analysis

## Troubleshooting

### Common Issues

**"No Data" Warning**
- Make sure you've loaded a simulation data file first
- Check that the file is a valid `.npz` file from your simulation

**Plot Errors**
- Verify that the simulation data contains the required fields
- Some plots require specific data (e.g., physics fields for pressure plots)
- Check the console output for detailed error messages

**Performance Issues**
- Large datasets may take time to load and plot
- Use the progress indicator to monitor loading status
- Consider reducing the number of time points for faster visualization

### Getting Help
- Check the console output for detailed error messages
- Ensure you're using compatible simulation data files
- Verify that all required dependencies are installed

## Technical Details

### Dependencies
- `tkinter` (usually included with Python)
- `matplotlib` for plotting
- `numpy` for data handling
- `pandas` for data export
- `PyYAML` for configuration files

### Data Requirements
The GUI expects simulation data files containing:
- `metadata`: Simulation parameters and configuration
- `field_data`: Cell density and nutrient fields
- `physics_data`: Physical fields (pressure, velocity, etc.)

### Configuration
User settings are stored in `~/.silicokit_gui_config.json`:
- Recent files list
- Window preferences
- Default parameters

## Integration with Existing Workflow

This GUI is designed to complement your existing simulation workflow:
- Run simulations as usual using `run_simulation.py`
- Use the GUI to quickly analyze results without switching to Jupyter
- Export plots and data for presentations or further analysis
- Keep the GUI open alongside your simulation runs for real-time monitoring

The GUI provides the same plotting capabilities as your Jupyter notebooks but in a more convenient, organized interface that's perfect for side-by-side analysis with running simulations.
