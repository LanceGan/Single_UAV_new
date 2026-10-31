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

## 2. A2G 原则（仅"其他论文的 baseline"不得使用 A2G）

- **我们的方法保留 A2G**：TD3、A*、MapGuide 都应使用 A2G（找高上传点、优化上传时延 τ），这是本工作的核心。
- **选自其他论文的 DRL baseline 不应有 A2G**：DQN、DDPG 原始论文没有 A2G，复现时应只给 G2A（状态里去掉 A2G SNR 维度），以忠实复现并体现 A2G 的价值。
- 序列 baseline 本来就 A2G-free：距离 TSP 只看距离、G2A-SA 只看距离+G2A 中断，已正确。

| 方法 | A2G | 说明 |
|---|---|---|
| TD3 / A* / MapGuide | ✅ 有 | 我们自己的方法 |
| 距离 TSP | ✅ 无 | 只看飞行距离（已正确） |
| G2A-SA | ✅ 无 | 只看距离 + G2A 中断（已正确） |
| **DDPG / DQN** | ❌ 待改 | 状态目前含 A2G SNR，需改成只含 G2A 的 7 维观察 |

## 3. 序列基线结果（已补齐）

- 距离 TSP 顺序 = G2A-SA 顺序 = **[1,6,5,2,3,4]**（G2A 中断不改变本场景的最优序列，5 个种子一致，且穷举验证为距离+中断的全局最优）。
- 距离 TSP 顺序在 MapGuide 轨迹下**全面优于 IGA 顺序**：TSP 264.5/365.5/488.5 vs IGA 273.0/369.0/492.0。
- A\* 下两者接近：TSP 216.5/317.0/445.5 vs IGA 214.0/318.0/447.0。
- 结论：数据感知 IGA 的经验优势有限；主贡献应放在 **TD3 连续轨迹优化**，序列作为次要消融如实报告。

## 4. Windows 已做的代码改动
（Mac 需知，且与 Mac 旧结果不兼容）

1. **奖励 v5**：上传奖励改为 `0.5 s/MB`（原来按"本航段数据量占比"归一化，数据量越大单步奖励越小，悬停上传几乎总是净负）。现在高效上传的净收益为正。
   - ⚠️ Mac 上 `formal-v7` 的所有结果用的是**旧奖励**。当前 Windows 代码是新奖励，两者不能混用；如果要重训，用新奖励重跑。
2. **示范概率**：
   - TD3/DDPG（`*_bc`）：`guide_min_prob=0.2`（前 400 回合从 0.5 降到 0.2，之后保持）。
   - D3QN：`guide_min_prob=0.5`（**常数，不衰减**，先确保能完成任务）。
   - 教师默认 MapGuide；`--teacher astar` 保留为可选消融。
3. 行为克隆权重：TD3/DDPG 前 600 回合 1.0 → 0.25 后保持。

## 5. 分支约定

| 分支 | 用途 | 当前指向 |
|---|---|---|
| `master` | **冻结，不更新** | `71b62b1` |
| `feat/win` | Windows 结果 + 代码改动 | `51956e4` |
| `feat/mac` | Mac 结果 | `083fb19` |

推送一律走 `git push new <分支>`（`new` = `Single_UAV_new`），不要碰 `origin`（旧仓库）。

## 6. 当前进行中

- **D3QN 重训（Windows）**：用**新奖励 + MapGuide 常数 0.5 引导**再试低负载。Mac 上 0/120 是旧奖励下的失败，不能据此判定新配方也失败；Windows 250 回合短训已出现正趋势（失败模式从"不传数据"变成"传了但慢"）。
- 命令：`D:/Anaconda/envs/pytorch/python.exe Train_D3QN.py --data_level low --teacher mapguide --seed 1 --episodes 3000 --output_root journal_results/d3qn-v3`

## 7. 待办 / 注意

- **DDPG/DQN 改 A2G-free（待确认）**：若需忠实复现，把这两个 DRL 基线状态里的 A2G SNR 去掉（改成 7 维 G2A-only 观察）并重跑；否则只在论文里说明其 G2A-only 的设定。
- **TD3 三档配方不完全一致**：低/中来自 Mac（0.2 下限配方），高来自 Win（旧 0 衰减配方）。论文里若要求完全一致，可考虑在 Windows 用 `guide_min_prob=0.2` 重训一版 high。
- 判断方法好坏一律看**纯 actor 的 `eval_best`**（无引导、无噪声），不看训练奖励曲线。
- 环境：Windows 用 `D:/Anaconda/envs/pytorch/python.exe`（Py3.11 + CUDA）；默认 `python` 是 3.14，没有 torch/pytest，不要用。
