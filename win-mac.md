# Win ↔ Mac 协同说明（2026-11-01）

本文件是 Windows 机与 Mac 机之间的交接约定，两边会话开始前先读这一份，保持一致。

## 1. 实验设计（三层，已定）

### 层面 1：轨迹方法对比（突出 TD3 优势）

- 固定：巡检顺序算法（IGA）+ A2G/G2A **全知**（双地图）。
- 变量：轨迹方法 = **TD3 vs D3QN vs A\***。
- 目的：所有条件相同下突出 TD3。

### 层面 2：QoS 与数据量敏感性

- 固定：IGA 序列 + 双地图全知 + **TD3**。
- 变量：QoS 阈值、数据量（低/中/高）。
- 看端到端指标与轨迹变化。

### 层面 3：其他论文 baseline 对比（A2G-free）

这些 baseline 的原始论文都没有 A2G；为忠实复现并突出 A2G 地图的价值，**它们一律不能考虑 A2G**：

1. Distance-TSP + TD3
2. DRL-SA
3. EEGA-TD3
4. SA-D3QN
5. Yang-style + TD3

## 2. A2G 原则（只适用于层面 3 的 baseline）

| 方法 | 所属层面 | A2G | 说明 |
|---|---|---|---|
| TD3（IGA 序） | 1、2 | ✅ 有 | 本文方法，双地图 |
| D3QN（IGA 序） | 1 | ✅ 有 | 与 TD3 的算法对比，同状态 |
| A\*（IGA 序） | 1 | ✅ 有 | oracle |
| MapGuide | 参照 | ✅ 有 | 我们的规划器 |
| **Distance-TSP + TD3** | 3 | ❌ 无 | 只距离 + A2G-blind TD3 |
| **DRL-SA** | 3 | ❌ 无 | 只 G2A + A2G-blind TD3 |
| **EEGA-TD3** | 3 | ❌ 无 | 待实现 |
| **SA-D3QN** | 3 | ❌ 无 | 待实现 |
| **Yang-style + TD3** | 3 | ❌ 无 | 待实现 |

DDPG 之前也失败（0/120），不进论文，仅内部知晓。

## 3. 结果分配（层面 1、2）

| 方法 | 低 T+τ (s) | 中 T+τ (s) | 高 T+τ (s) | 来源 |
|---|---|---|---|---|
| **TD3** | 235.5（97/120） | 351.5（73/120） | 501.0（39/120） | 低/中 → Mac；高 → Win |
| A\* | 214.0 | 318.0 | 447.0 | Win |
| D3QN | 失败 0/120 | — | — | 负结果，一句话带过 |

## 4. 分支约定

| 分支 | 用途 |
|---|---|
| `master` | **冻结，不更新** |
| `feat/win` | Windows 结果 + 代码改动 |
| `feat/mac` | Mac 结果 + Mac 领先的代码 |

- 推送走 `git push new <分支>`（`new` = `Single_UAV_new`），不要碰 `origin`（旧仓库）。
- `journal_results/**/models/` 与 `logs/` 不入库；数字都在 `eval_history.json` / `*.npz` / `paper_figures/`。

## 5. Windows 已做的代码改动

1. **奖励 v5**：上传奖励改为 `0.5 s/MB`（不再按航段数据量归一化），高效上传净收益为正。
2. **示范概率**：TD3/DDPG（`*_bc`）`guide_min_prob=0.2`；D3QN `guide_min_prob=0.5`（常数）。
3. **World** 新增 `include_a2g_snr` 参数 + `state_dim_for()`：关闭时状态去 A2G SNR（8→7 维），供层面 3 的 A2G-blind 实验复用（上传奖励保留）。
4. 行为克隆权重：TD3/DDPG 1.0 → 0.25 下限。

## 6. 当前进行中

- **D3QN 正确配方重跑中（Windows）**：A2G-aware 8 维 + guide 0.5 常数 + 新奖励，3000 轮，输出 `journal_results/d3qn-v4`。此前误设的 A2G-free 版 0/120 不算数，以这次为准。
- **Mac**：Distance-TSP+TD3 与 DRL-SA（A2G-blind）训练中。

## 7. 待办

- [ ] 层面 1：D3QN 正确配方结果确认（Win，进行中）
- [ ] 层面 2：QoS 扫描（TD3，不同 T_out 阈值）
- [ ] 层面 3 未实现 baseline：**EEGA-TD3、SA-D3QN、Yang-style+TD3**
- [ ] 课程版 Proposed（IGA+TD3 low，Mac 在等，需 ≤229.0s）
- [ ] 配方口径定稿：`0.5→0`（course-final）vs `0.5→0.2`（legacy-0.2）
- [ ] 判断好坏一律看纯 actor `eval_best`（无引导、无噪声）
