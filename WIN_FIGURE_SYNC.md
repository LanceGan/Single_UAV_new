# V5 待补图同步（Win → Mac）

更新时间：2026-11-09

## 结论

待补图**并非全部由 Win 负责**：

| 待补图 | 数据来源 | 负责端 |
|---|---|---|
| E1 Low：CA-TD3、D3QN、PPO、DDPG、SAC 的 A2G/G2A 轨迹 | CA-TD3、D3QN 模型在 Mac；PPO、SAC、DDPG 的 `eval_best` actor 未随结果包推送 | Mac + Win |
| E5 Low：七种方法的统一轨迹图 | MCO-A* 轨迹和其余 baseline actor 均在 Mac | Mac |
| E4 Medium/High：代表 QoS/权重点的轨迹面板 | 12 组 V5 训练的 actor 均在 Mac | Mac |

请 Win 只补 E1 Low 中 **PPO、SAC 和 DDPG** 的轨迹数据；D3QN 与
CA-TD3 由 Mac 完成。

## Win 需要补的文件

请从对应训练 run 的 `models/` 导出两个神经网络的 actor，并执行一次与训练
完全一致的确定性评估。不要使用带探索噪声或 MapGuide 代飞的轨迹。

### 1. PPO Low

- run：
  `journal_results/v5-win/E1/ppo/A/low/ppo`
- `eval_best.episode = 900`
- 预期指标：`T=190.0 s`、`tau=75.5 s`、`T_out^cont=3.0 s`、
  6/6 巡检、450/450 MB、QoS 可行

### 2. SAC Low

- run：
  `journal_results/v5-win/E1/sac/A/low/sac`
- `eval_best.episode = 925`
- 预期指标：`T=170.0 s`、`tau=81.5 s`、`T_out^cont=4.0 s`、
  6/6 巡检、450/450 MB、QoS 可行

### 3. DDPG Low（失败诊断轨迹，可选但建议提供）

- run：
  `journal_results/v5-win/E1/ddpg/baseline/low/ddpg_bc`
- 训练选择器的诊断 episode：`1750`
- 该次只完成 4/6、300/450 MB，`T_out^cont=20.0 s`
- **没有有效 `T+tau` 目标**；图上必须明确标注 `incomplete / infeasible`

如果 DDPG 的该诊断 actor 没有保存，不要用最终 actor 冒充该 episode；请
在 MANIFEST 中明确写 `unavailable`。

## 请推送的目录结构

在 `feat/win` 分支新增：

```text
sync_bundles/v5_e1_low_trajectories/
├── MANIFEST.json
├── ppo_actor_eval_best.pth
├── sac_actor_eval_best.pth
├── ddpg_actor_eval_best.pth       # 如可得
├── e1_low_ppo_trajectory.npz
├── e1_low_sac_trajectory.npz
├── e1_low_ddpg_trajectory.npz     # 如可得
├── e1_low_ppo_metrics.json
├── e1_low_sac_metrics.json
└── e1_low_ddpg_metrics.json       # 如可得
```

每个 `trajectory.npz` 至少包含：

```text
x, y, time_s, step
```

若有 `inspection_order`、访问点时间和累计上传量，也一并保存。每条轨迹必须
使用 Low payload、6 s 最长连续中断约束、seed 1、PRIGA 顺序
`[1,6,2,3,5,4]`，且评估时关闭探索噪声与 MapGuide。

`MANIFEST.json` 需要记录：

- 源码 commit；
- `run_config.json` 路径；
- actor 文件路径与 SHA-256；
- 网络类型及加载方式；
- `eval_best.episode`；
- 重放后的 `T`、`tau`、`T_out^cont`、访问点数、上传量、是否可行；
- 是否与 `eval_history.json` 的 eval-best 完全一致。

## 验收标准

Mac 收到后会逐项核对：

1. PPO、SAC 的 `T/tau/T_out` 必须分别与 900、925 episode 的记录一致；
2. 必须为 6/6 巡检及 450/450 MB；
3. DDPG 只作为失败诊断展示，不能出现在可行方法排名中；
4. 若 actor 无法恢复，必须明确返回 `unavailable`，不能用 final actor 或无
   eval-best 标签的模型替代；
5. 通过后由 Mac 统一合成五执行器 A2G/G2A 对比图，避免 Win/Mac 使用不同
   地图和样式。

完成推送后请告知 commit hash。
