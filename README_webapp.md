# Geometric Z-Prime Web Application

This Flask web application provides an interactive interface for visualizing prime number distributions using geometric projections based on the Universal Frame Shift theory.

## Features

- **Interactive Parameter Tuning**: Adjust all parameters through a user-friendly web interface
- **Real-time Visualization**: Generate plots dynamically with the "Draw" button
- **Input Validation**: Comprehensive validation with helpful error messages
- **Responsive Design**: Modern, responsive interface that works on various screen sizes
- **Statistical Analysis**: Real-time analysis results and prime candidate detection

## Parameters

### Analysis Parameters
- **Range (N)**: Maximum number to analyze (1K - 100K)
- **Exponent (K)**: Z scaling exponent for frame shift calculations
- **Density Threshold**: Number of top density points to consider

### Frequency Parameters
- **Base Frequency**: Base frequency multiplier for sine wave generation
- **Earth Omega**: Earth's rotational frequency (rad/s) for cosmic tuning
- **Nearest K**: Number of nearest neighbors for local density calculation

### Visualization Parameters
- **Transparency (Alpha)**: 3D scatter point transparency (0.0 - 1.0)
- **3D Point Size**: Size of 3D scatter points
- **2D Point Size**: Size of 2D scatter points/markers
- **Elevation**: 3D plot elevation angle (-90° to 90°)
- **Azimuth**: 3D plot azimuth angle (-180° to 180°)

## Usage

1. Start the Flask application:
   ```bash
   python webapp.py
   ```

2. Open your browser and navigate to `http://localhost:5000`

3. Adjust parameters in the sidebar

4. Click "Draw" to generate the visualization

5. View the results in the main area, including:
   - 3D geometric projection plot
   - 2D density analysis plot
   - Statistical summary
   - Prime candidate list

## Technical Details

- **Backend**: Flask web framework with matplotlib for plotting
- **Frontend**: HTML/CSS/JavaScript with responsive design
- **Plotting**: Matplotlib with Agg backend for web compatibility
- **Data Processing**: NumPy and SciPy for numerical computations
- **Visualization**: Base64-encoded PNG images for web display

## Error Handling

The application includes comprehensive error handling:
- Input validation with range checking
- Exception handling for computation errors
- User-friendly error messages
- Graceful degradation for invalid inputs

## Performance

Optimized for web performance:
- Limited maximum range to 100K numbers for reasonable response times
- Efficient plotting with appropriate point sizes
- Compressed image output for faster loading
- Asynchronous form submission with loading indicators