# Beyond the Thermostat: Advancing Building Automation with Thermal Digital Twins

**12-770 Research Project — Carnegie Mellon University**

A research project building thermal digital twins for real indoor environments using physics-informed neural operators trained on real wireless sensor data.

## Project Overview

We train [DeepONet](https://arxiv.org/abs/1910.03193) and PINN-based operator networks on real temperature/humidity sensor measurements from two university labs, with the goal of learning the underlying thermodynamic physics — not just fitting a curve.

Core research questions:
- Can we extract the physics a DeepONet actually learned from real (noisy, irregular) sensor data?
- Does a model trained on Lab 1 generalise zero-shot to Lab 2?
- How do operator architectures (DeepONet, FNO, LSTM-PINN) compare on real indoor data?

## Progress Journal

The GitHub Pages site documents progress along the way:
[https://nuralik.github.io/Beyond-the-Thermostat-Advancing-Building-Automation-with-Thermal-Digital-Twins/](https://nuralik.github.io/Beyond-the-Thermostat-Advancing-Building-Automation-with-Thermal-Digital-Twins/)

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Project Structure

```
├── pyproject.toml        # Project metadata and dependencies (uv)
├── data/                 # Sensor data (temperature, humidity, RSSI)
├── deepOnet/             # DeepONet training and experiments
├── pinn/                 # Physics-Informed Neural Network experiments
├── lstm/                 # LSTM-PINN baseline
├── data_analysis/        # Exploratory analysis notebooks
├── index.html            # GitHub Pages — research questions
└── progress.html         # GitHub Pages — progress journal
```

## Data

Real wireless sensor data from two instrumented labs (Lab 1 and Lab 2), including temperature, relative humidity, and RSSI measurements at irregular sensor locations. Dataset originally published in [this paper](data-08-00082-v2.pdf).

## Course

CMU 12-770 · Spring 2026
