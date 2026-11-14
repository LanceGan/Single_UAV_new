# [ICC2026]Dual Radio Map-Aware Flight Strategy Optimization for UAV-Based Inspection System

## 📖 Overview

This repository contains the implementation of a cellular-connected UAV inspection system that optimizes both flight trajectory and inspection sequence using dual radio maps. The system employs a hybrid algorithm combining Improved Genetic Algorithm (IGA) and Deep Reinforcement Learning (DRL) to minimize mission completion time and data transmission delay in rural environments.

## 🖼 论文图与绘图代码对应关系

下表对应当前论文仓 `P3` 的 `journal.tex` 中 11 张编号图。论文仓默认与本仓库位于同一父目录，因此下表中的 `../P3/...` 可直接使用。

| 论文图 | 内容 | 论文引用文件 | 绘图/生成代码 | 主要源数据 |
|---|---|---|---|---|
| Fig. 1 | 双无线电地图 UAV 巡检系统示意图 | `../P3/figs/system_model.pdf` | 当前 PDF 由 PowerPoint 导出，论文仓没有保存对应的 PowerPoint 或可原样重生成的代码；`../P3/figs/system_model.tex` 是可编译的旧版 TikZ 底稿，但输出尺寸与当前 PDF 不同 | 当前 PDF 与旧版 TikZ 源码 |
| Fig. 2 | DRM-HO 框架图 | `../P3/figs/DRM-HO.png` | 当前成图没有找到完全一致的已跟踪生成脚本；仓库维护版生成器为 `plot_algorithm_framework.py`，输出 `paper_figures/algorithm_framework/drm_ho_framework.*` | `plot_algorithm_framework.py` 内的示意图定义 |
| Fig. 3 | E1 训练奖励曲线 | `../P3/figs/v5/e1_reward_curves_v5_panel_{a,b,c}.pdf` | `plot_v5_e1_training_validation.py` | `../P3/figs/v5/source_data/e1_training_rewards_v5.csv`, `e1_training_validation_v5.json` |
| Fig. 4 | E1 确定性可行率曲线 | `../P3/figs/v5/e1_feasibility_curves_v5_panel_{a,b,c}.pdf` | `plot_v5_e1_training_validation.py` | `../P3/figs/v5/source_data/e1_feasibility_v5.csv`, `e1_training_validation_v5.json` |
| Fig. 5 | E1 四执行器在 A2G SNR 地图上的轨迹 | `../P3/figs/v5/e1_a2g_trajectories_v5_panel_{a,b,c,d}.pdf` | `plot_v5_e1_all_trajectories.py` | `../P3/figs/v5/source_data/e1_payload_trajectories_v5.{npz,csv,json}`, `e1_trajectory_results_v5.csv` |
| Fig. 6 | 同一批执行器在 G2A 覆盖概率地图上的轨迹 | `../P3/figs/v5/e1_g2a_trajectories_v5_panel_{a,b,c,d}.pdf` | `plot_v5_e1_all_trajectories.py` | `../P3/figs/v5/source_data/e1_payload_trajectories_v5.{npz,csv,json}`, `e1_trajectory_results_v5.csv` |
| Fig. 7 | E3 A2G 可见性消融轨迹 | `../P3/figs/v5/e3_a2g_trajectories_v5_panel_{a,b,c}.pdf` | `plot_v5_e3_ablation_trajectories.py` | `../P3/figs/v5/source_data/e3_a2g_ablation_v5.json`, `e3_a2g_trajectories_v5.{npz,csv,json}` |
| Fig. 8 | E2 巡检顺序算法轨迹多样性 | `../P3/figs/v5/e2_inspection_order_trajectories_v5_panel_{a,b,c,d,e,f}.pdf` | `plot_v5_inspection_order_trajectories.py` | `../P3/figs/v5/source_data/e2_inspection_order_trajectories_v5.{npz,csv,json}`, `e2_trajectory_panels_v5.json` |
| Fig. 9 | E4 Low 档 QoS 与权重轨迹泛化 | `../P3/figs/v5/e4_low_qos_trajectories_v5_panel_a.pdf`, `e4_low_weight_trajectories_v5_panel_b.pdf` | `../P3/figs/v5/plot_e4_payload_generalization.py` | `../P3/figs/v5/source_data/e4_qos_trajectories_v5.{json,npz}`, `e4_weight_trajectories_v5.{json,npz}`, `e4_low_generalization_trajectories_v5.{csv,json}` |
| Fig. 10 | E5 Low 档实时中断概率与连续中断时长 | `../P3/figs/v5/e5_low_realtime_v5_panel_{a,b}.pdf`, `e5_low_realtime_v5_legend.pdf` | `plot_v5_e5_time_series.py` | `../P3/figs/v5/source_data/e5_low_realtime_v5.{npz,csv,json}` |
| Fig. 11 | E5 八个端到端算法的 Low/Medium/High 轨迹 | `../P3/figs/v5/e5_all_level_trajectories_v5_panel_{a,b,c,d,e,f,g,h}.pdf` | `../P3/figs/v5/plot_e5_all_level_trajectories.py` | `../P3/figs/v5/source_data/e5_all_level_trajectories_v5.{npz,csv,json}` |

对应的复现命令：

```bash
# 在 Single_UAV_new 根目录运行。
python3 plot_algorithm_framework.py
python3 plot_v5_e1_training_validation.py
python3 plot_v5_e1_all_trajectories.py
python3 plot_v5_e3_ablation_trajectories.py
python3 plot_v5_inspection_order_trajectories.py
python3 plot_v5_e5_time_series.py

# Fig. 9 的脚本位于论文仓。
cd ../P3
python3 figs/v5/plot_e4_payload_generalization.py

# Fig. 11 的脚本位于论文仓，但必须从实验仓根目录运行。
cd ../Single_UAV_new
python3 ../P3/figs/v5/plot_e5_all_level_trajectories.py

# Fig. 1 的旧版 TikZ 底稿可独立编译；它不是当前 PowerPoint PDF 的原样生成器。
cd ../P3
cd figs
pdflatex system_model.tex
```

注意：

- `plot_v5_inspection_order_trajectories.py` 会重新执行多种顺序对应的确定性回放，运行时间明显长于其余静态绘图脚本。
- `plot_v5_e1_all_trajectories.py`、E3/E5 脚本依赖本仓库中的 checkpoint、同步 bundle 和实验结果目录；只复制绘图脚本无法独立重画。
- Fig. 1 当前的 `system_model.pdf` 由 Microsoft PowerPoint 2021 导出，不是由 `system_model.tex` 生成；旧 TikZ 文件只能用于早期版式复现。
- Fig. 2 当前论文资产与维护版生成器之间存在版式差异。若要用代码重新生成 Fig. 2，应先比较 `drm_ho_framework.*` 与 `DRM-HO.png`，确认新版内容后再替换论文资产。

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

论文图的完整生成映射和命令见上文“论文图与绘图代码对应关系”。实验训练入口、评价入口和 V5 矩阵说明见仓库中的 `V5_EXPERIMENT_ASSIGNMENT.md`、`V5_IMPLEMENTATION_STATUS.md` 和 `V5_FINAL_MATRIX.md`。

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
