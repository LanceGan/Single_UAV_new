# 正式实验论文图设计

## 目的

为 formal-v5-restart 的四组3000轮实验分别生成可直接用于论文分析的轨迹图、优化指标曲线和奖励曲线。

## 输出

每个运行目录下创建 `paper_figures/`：

- `trajectory_eval_best_dual_maps.png/.pdf`：eval_best纯Actor轨迹叠加G2A覆盖概率和A2G SNR地图。
- `objective_vs_episode.png/.pdf`：每25轮评估的 T、tau_total 和 T+tau_total；失败评估用叉号/低透明度显示，可行目标值单独强调。
- `reward_vs_episode.png/.pdf`：3000轮训练回报、200轮移动平均以及纯Actor评估回报。
- `figure_summary.json`：图中采用的最佳轮次、成功次数和最佳指标。

DDPG没有可行评估时仍绘制完整失败曲线与轨迹，不伪造可行目标值。所有图片同时输出300 DPI PNG和矢量PDF。

