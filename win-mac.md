# Win ↔ Mac 协同说明（2026-10-31）

本文件是 Windows 机与 Mac 机之间的交接约定，两边会话开始前先读这一份，保持一致。

## 1. 论文结果分配（已定）

主方法 TD3 的三档结果分别来自两台机器：

| 方法 | 低负载 T+τ (s) | 中负载 T+τ (s) | 高负载 T+τ (s) | 来源 |
|---|---|---|---|---|
| **TD3** | **235.5**（97/120） | **351.5**（73/120） | **501.0**（39/120） | 低/中 → Mac；高 → Win |
| A* | 214.0 | 318.0 | 447.0 | Win（`astar-fixed`，逐段最优，World 重放） |
| MapGuide | 273.0 | 369.0 | 492.0 | Win（Dijkstra + 上传点启发式） |

- **失败结果不进论文**，仅内部知晓：DDPG 0/120、D3QN 0/120（Mac 上 A* 示范对与 MapGuide 整回合两种配方都失败）。
- **TD3 high** 用 Win 的 `formal-v5-restart` 结果（Mac 的 high 只有 1/120，已弃用）。

## 2. Windows 已做的代码改动（Mac 需知，且与 Mac 旧结果不兼容）

1. **奖励 v5**：上传奖励改为 `0.5 s/MB`（原来按"本航段数据量占比"归一化，数据量越大单步奖励越小，悬停上传几乎总是净负）。现在高效上传的净收益为正。
   - ⚠️ Mac 上 `formal-v7` 的所有结果用的是**旧奖励**。当前 Windows 代码是新奖励，两者不能混用；如果要重训，用新奖励重跑。
2. **示范概率**：
   - TD3/DDPG（`*_bc`）：`guide_min_prob=0.2`（前 400 回合从 0.5 降到 0.2，之后保持）。
   - D3QN：`guide_min_prob=0.5`（**常数，不衰减**，先确保能完成任务）。
   - 教师默认 MapGuide；`--teacher astar` 保留为可选消融。
3. 行为克隆权重：TD3/DDPG 前 600 回合 1.0 → 0.25 后保持。

## 3. 分支约定

| 分支 | 用途 | 当前指向 |
|---|---|---|
| `master` | **冻结，不更新** | `71b62b1` |
| `feat/win` | Windows 结果 + 代码改动 | `f097cd4` |
| `feat/mac` | Mac 结果 | `083fb19` |

推送一律走 `git push new <分支>`（`new` = `Single_UAV_new`），不要碰 `origin`（旧仓库）。

## 4. 当前进行中

- **D3QN 重训（Windows）**：用**新奖励 + MapGuide 常数 0.5 引导**再试低负载。Mac 上 0/120 是旧奖励下的失败，不能据此判定新配方也失败；Windows 250 回合短训已出现正趋势（失败模式从"不传数据"变成"传了但慢"）。
- 命令：`D:/Anaconda/envs/pytorch/python.exe Train_D3QN.py --data_level low --teacher mapguide --seed 1 --episodes 3000 --output_root journal_results/d3qn-v3`

## 5. 待办 / 注意

- **TD3 三档配方不完全一致**：低/中来自 Mac（0.2 下限配方），高来自 Win（旧 0 衰减配方）。论文里若要求完全一致，可考虑在 Windows 用 `guide_min_prob=0.2` 重训一版 high。
- 判断方法好坏一律看**纯 actor 的 `eval_best`**（无引导、无噪声），不看训练奖励曲线。
- 环境：Windows 用 `D:/Anaconda/envs/pytorch/python.exe`（Py3.11 + CUDA）；默认 `python` 是 3.14，没有 torch/pytest，不要用。
