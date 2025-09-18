# Tuning_sim2

**Quantexa Scorecard Simulation and Optimization Project**

This project simulates data and a Quantexa scorecard, then trials different optimization methods against it. This is a fresh restart focused on cleaner, less repetitive code compared to the original tuning_sim project.

## Project Overview

The project creates simulated financial data with configurable scorecards and binary outcomes, then evaluates different machine learning approaches for predicting these outcomes. The goal is to understand how different optimization methods perform against known scorecard structures.

## Key Objectives

- Implement and evaluate a Quantexa scorecard on simulated data
- Trial different optimization methods for tuning the scorecard
- Analyze how different parameter configurations affect scorecard performance

## Setup

This project uses UV for dependency management. To get started:

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Or use UV commands directly:
   ```bash
   uv run python script.py
   uv run jupyter notebook
   ```

## Project Structure

- `code/`: Python modules and scripts
- `notebooks/`: Jupyter notebooks for analysis
- `data/`: Raw data files
- `output/`: Generated results and visualizations
- `memory-bank/`: Project documentation and progress tracking

## Memory Bank

This project follows the Memory Bank approach for documentation:
- See `memory-bank/project_context.md` for project overview and technical details
- See `memory-bank/progress.md` for current status and progress
- See `.clinerules` for development methodology and workflow principles
