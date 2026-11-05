# [ICC2026]Dual Radio Map-Aware Flight Strategy Optimization for UAV-Based Inspection System

## 📖 Overview

This repository contains the implementation of a cellular-connected UAV inspection system that optimizes both flight trajectory and inspection sequence using dual radio maps. The system employs a hybrid algorithm combining Improved Genetic Algorithm (IGA) and Deep Reinforcement Learning (DRL) to minimize mission completion time and data transmission delay in rural environments.

## 🎯 Key Features

- **Dual Radio Map Integration**: Leverages both Ground-to-Air (G2A) and Air-to-Ground (A2G) radio maps for comprehensive communication awareness
- **Hybrid Optimization**: Combines IGA for discrete inspection sequence planning and DRL for continuous trajectory optimization
- **Real-time Data Transmission**: Ensures reliable uplink data offloading while maintaining stable downlink connectivity
- **Rural Environment Adaptation**: Uses 3GPP Rural Macro (RMa) channel model for realistic propagation characteristics

## 🏗 System Architecture

The system consists of:
- **Cellular-connected UAV** with high-resolution camera for inspection tasks
- **Multiple GBS sectors** providing cellular connectivity
- **Pre-constructed radio maps** (G2A and A2G) guiding communication-aware path planning
- **Mission-critical inspection points** requiring real-time data transmission

## ⚡ Problem Formulation

The joint optimization problem minimizes:

Where:
- **T**: Total mission completion time
- **τ_total**: Total transmission delay

### Constraints:
- Single sector association per time step
- Complete data offloading between inspection points
- Maximum continuous outage duration limit
- UAV kinematic constraints
- Boundary conditions

## 🧠 Algorithm: DRL-IGA

### Phase 1: Inspection Sequence Optimization (IGA)
- Models the problem as a modified Traveling Salesman Problem (TSP)
- Defines composite edge weights considering:
  - Flight distance
  - Uplink communication throughput
  - Downlink outage duration
- Uses genetic algorithm for near-optimal sequence determination

### Phase 2: Flight Trajectory Optimization (DRL)
- Employs Twin Delayed Deep Deterministic Policy Gradient (TD3)
- State space: UAV position, data buffer, target coordinates
- Action space: Velocity and heading angle
- Reward function balancing mission completion and communication quality

## 📊 Performance Results

The proposed DRL-IGA demonstrates significant improvements:

- **Up to 20% reduction** in total task completion time and transmission delay compared to baseline methods
- **Balanced optimization** of uplink data rates and downlink outage duration
- **Effective avoidance** of low-coverage regions while maintaining efficient flight paths

### Comparison with Baseline Methods:
- **TSP**: Considers only flight distance, ignores radio maps
- **DRL-SA**: Uses simulated annealing with G2A map only
- **DRL-IGA (Ours)**: Jointly utilizes both G2A and A2G radio maps

## 🔧 Implementation Details

### Simulation Environment:
- Area: 2km × 2km rural environment
- 4 GBSs with 12 total sectors
- 6 inspection points
- 3GPP RMa channel model
- Building distribution with realistic obstacles

### Communication Parameters:
- Carrier frequency: 2 MHz
- Bandwidth: 1 MHz
- UAV transmit power: 0.1 W
- GBS transmit power: 0.15 W
- Maximum continuous outage duration: experiment-dependent (legacy cumulative budget was 80s)

## 🚀 Getting Started

[Code implementation details and usage instructions would be added here when the code is released]

## 📋 Requirements

[List of dependencies and requirements would be added here]

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{gan2024dual,
  title={Dual Radio Map-Aware Flight Strategy Optimization for UAV-Based Inspection System},
  author={Gan, Ruijie and Peng, Haixia and Cao, Jiangling and Su, Zhou and Luan, Tom H and Cheng, Nan},
  conference={IEEE ICC 2026},
  year={2026}
}
