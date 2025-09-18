# Project Context

## Project Overview
Project to simulate data and a Quantexa scorecard, then trial different optimization methods against it. This is a fresh restart of the original tuning_sim project, focused on creating cleaner, less repetitive code while maintaining core functionality.

The project centers around creating realistic financial data simulations with configurable scorecard structures and binary outcomes, then evaluating how well different machine learning approaches can predict these outcomes when the underlying scorecard structure is known.

## Data Sources
Only simulated data - no external data sources required. All data is generated through the Scorecard class simulation system.

## Key Questions
- How can we implement and evaluate a Quantexa scorecard on simulated data?
- What optimization methods are most effective for tuning the scorecard?
- How do different parameter configurations affect scorecard performance?

## Tech Stack & Dependencies
- Data manipulation: pandas, numpy
- Visualization: matplotlib, seaborn, plotly
- Machine learning: scikit-learn, scipy
- Graph analysis: networkx
- Utilities: re, collections, beautifulsoup4, arrow (dates/times), faker
- Notebook support: jupyter, ipykernel
- Development tools: pytest, black, flake8

## Environment Setup
- Use the UV virtual environment: `source .venv/bin/activate`
- Or use UV commands directly: `uv run python script.py` or `uv run jupyter notebook`
- Install additional dependencies: `uv add package_name`
- The UV environment includes project /code in the Python path

## Module Import Strategy
Import modules directly without the 'code.' prefix:
```python
from data_simulation import simulate_data
```
Do NOT use 'code.' prefix in imports.

## Code Organization Principles
1. **Separate Analysis Code from Presentation**:
   - Analysis logic in Python scripts (code/ directory)
   - Notebooks focused on reviewing results and visualization

2. **Modular Code Structure**:
   - Separate Python modules for different analysis components
   - Import modules into notebooks for result presentation

3. **Scorecard Simulation Foundation**:
   - Core Scorecard class for generating realistic financial features
   - Configurable weights and binary outcome generation
   - Clean API with proper type hints and comprehensive testing

4. **Notebook Output Management**:
   - Always clear notebook outputs before making changes: `python clear_notebook_outputs.py notebooks/`
   - Clean outputs after completing analysis work to prevent version control bloat
   - Use the clear_notebook_outputs.py script to remove execution counts and kernel metadata
   - This ensures clean diffs and prevents repository size growth from output data

## Results and Insights
This is a restart project - results will be generated as functionality is rebuilt with cleaner code patterns.

## Project-Specific Guidelines
- Implement only essential functionality for the current task
- Use simplest implementation that meets requirements  
- Focus on readability and maintainability
- Start with core Scorecard simulation, then add analysis capabilities incrementally
- Do not create complex analysis logic in notebooks
- Do not mix analysis code with visualization code
