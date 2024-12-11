# Project README

## Overview

This repository contains scripts for simulating and analyzing various scenarios, including synthetic data generation, real-world data analysis, and a "Deal or No Deal" game simulation. Below are instructions on how to set up the project, run the scripts, and visualize the results.

## Setup Instructions

To set up the project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**:
   Make sure you have `requirements.txt` in your repository. Install the necessary packages using:
   ```bash
   pip install -r requirements.txt
   ```

## Running `synthetic.py` and Plotting Results

To run the synthetic data generation script and plot the results, follow these steps:

1. **Run the Script**:
   ```bash
   python synthetic.py
   ```

2. **Plot the Results**:
   You can use `synthetic_plot.ipynb` to generate and save plots. 

## Running `realworld.py` and Plotting Results

To analyze real-world data and plot the results, do the following:

1. **Run the Script**:
   ```bash
   python realworld.py
   ```

You can also use the `script.sh` to run the code using nohup and redirect the output to a file.

## Deal or No Deal Simulation

The "Deal or No Deal" part of the project simulates the game mechanics for the deal or no deal benchmark.
