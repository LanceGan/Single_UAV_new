# Mac–Win 双机同步备忘

> 维护：Mac 侧（每次 Mac 有新结果/新约定时更新本文件并推送 `feat/mac`）
> 最后更新：2026-11-01
> 目的：两台机器并行做实验，用此文件对齐"谁在跑什么、哪份结果算数、代码在哪个分支"，避免重复劳动和口径混乱。

---

## 1. 分支约定

| 分支 | 内容 | 维护方 |
|---|---|---|
| `master` | 代码主线 + 交接/计划文档 | Win 侧推送，Mac 侧 rebase 跟进 |
| `feat/mac` | Mac 机的**全部实验产物** + Mac 侧领先于 master 的代码提交 | Mac 侧 |
| （待建）旧机结果分支 | Win/原机的实验产物（formal-v5 等） | Win 侧 |

规则：
- **结果只进本机分支**，不直接进 master；代码修改最终应合入 master（Mac 侧代码提交目前在 feat/mac 上领先，Win 侧可随时 merge）。
- `journal_results/**/models/`（训练 checkpoint，~814MB）与 `journal_results/**/logs/`（TensorBoard 原始流，单文件最大 129MB **超 GitHub 100MB 硬限制**）不入库；分析所需数字全部在 `eval_history.json` / `*.npz` / `paper_figures/` 中。
- Mac 侧 push 走 SSH（HTTPS 对 github 不稳定）；Win 侧 push 需开代理 `127.0.0.1:7897`，且注意 Win 机的 `origin` 指向旧仓库 SingleUAV，**推这个仓库要用 `git push new <branch>`**。

## 2. 配方注册表（看结果前先对配方）

| 代号 | 教师 | 教师概率 | BC 权重 | 其他 | 状态 |
|---|---|---|---|---|---|
| **course-final（课程正式）** | MapGuide | 0.5 → **0**（400 轮归零） | 1.0→0.25 下限 | D3QN 额外 `--learning_rate 1e-4` | ✅ 课程报告唯一口径 |
| legacy-0.2（旧配方） | MapGuide | 0.5 → **0.2** 持久下限 | 同上 | — | 仅作对照/期刊线历史数据 |
| A\* 教师（已废弃） | A\* 固定轨迹回放 | — | — | — | ❌ 实证失败：无噪声、不进回放的固定示范教不会任何 DRL（TD3 卡死 2 点平台；D3QN 最佳策略=训练前网络）。留档勿复用 |

## 3. Mac 侧结果清单（feat/mac 分支内）

### 课程线（low，course-final 配方）✅ 可直接进课程表

| 方法 | 顺序 | 成功率 | eval_best T+τ | 目录 |
|---|---|---|---|---|
| Distance-TSP+TD3 | `[1,6,5,2,3,4]`（720 排列精确） | 91/120 | **229.0s**（ep1525，T=140.0，τ=89.0，T_out=23.5） | `journal_results/course-baselines/distance-tsp/A/low/td3_bc/` |
| DRL-SA（SA 序 + TD3 轨迹） | `[1,6,5,2,3,4]`（SA 仅看 G2A） | 91/120 | **229.0s**（与上行逐位相同） | `journal_results/course-baselines/drl-sa/A/low/td3_bc/` |

两行相同的原因：SA 与距离 TSP 在本地图恰好选出同一顺序，加之同种子同配方 → 训练为确定性复制。课程表照填两行并脚注说明。SA 轨迹部分为统一 TD3（原文 DRL-SA 用 DQN，报告按 BASELINE_COMPARISON_PLAN §1 注明适配）。

**顺序质量核查**（`journal_results/analysis-astar-order-check/`）：A\* 规划器下 IGA 序 `[1,6,2,3,5,4]` = 214.0s < 基线序 216.5s → 数据感知顺序在低档亦严格更优。

### 期刊线 / 对照（legacy-0.2 配方）

| 方法 | 档位 | 结果 | 状态 |
|---|---|---|---|
| TD3（IGA 序） | low | 97/120，235.5s（ep2450） | 期刊线参考；课程 Proposed 行**不用它**（配方旧） |
| TD3（IGA 序） | medium | 73/120，351.5s（ep2625） | 期刊线；末期退化属已知现象，eval_best 已锚定 |
| TD3（IGA 序） | high | 1/120，550.0s | ❌ 弃用，期刊高档取 Win 侧 formal-v5（39/120，501.0） |
| DDPG | low | 0/120（A\* 教师版） | 失败结论成立；课程表口径见 §4 |
| D3QN | low | 0/120（A\* 示范对版与 MapGuide 整回合版各一次） | 失败结论成立（Win 侧独立复现） |

### 留档（勿当正式结果引用）

- `formal-v7/A/low/td3_bc_astar-teacher-failed/`、`d3qn-baseline/low/d3qn_astar-teacher-failed/`：A\* 教师失败实验的完整记录。

## 4. 跨机依赖（Win 侧欠 Mac 侧的 / 反之）

**Mac 等 Win：**
1. **课程版 Proposed**（IGA+TD3，low，course-final 配方）的产物——课程表的主行，且必须 ≤229.0s 才能闭环"Proposed 最优"的主张（IGA 序的 A\* 上界 214.0 支持这一点，若单种子没压过 229 需要加种子讨论）；
2. 期刊高档：formal-v5 高档产物（39/120，501.0）随旧机结果分支同步；
3. DDPG 课程表口径确认（建议引用 formal-v5 的 MapGuide 版 0/120，与其他方法同教师）。

**Win 可取用 Mac（都在 feat/mac）：**
- 代码：CUDA 可用性保护、D3QN MapGuide 教师（`--teacher mapguide`）、`--sequence_method {iga,distance_tsp,sa}`、绘图工具链（§5）。
- A\* 教师缓存 `journal_results/astar-teacher/`（三档已生成，目标值与 Win 侧一致：214.0/318.0/447.0）。

## 5. 图表与命名规范（两侧统一）

- 图例/标题一律 **TD3 / DDPG / D3QN**，不带 "-BC"（代码已强制：`evaluate_policy.method_display_name`）；顺序基线图题自动加前缀（`plot_formal_results.method_title`）。
- 奖励曲线展示版：**0–2000 轮、仅 actor 回合、200 回合移动平均、保留失败区间**（BASELINE_COMPARISON_PLAN §13；工具 `plot_rewards_display` / `plot_algorithm_comparison.py --run LABEL=DIR`）。
- 失败回合不得以原始终止值出现在数值轴上（对比图用顶部存在带表示不可行评估）。
- 结果一律报 **纯 actor `eval_best`**；提前终止/QoS 失败/超时不参与可行排名。

## 6. 快查数字（low，截至本文更新）

| 方法 | T+τ (s) | 备注 |
|---|---|---|
| A\*（IGA 序） | 214.0 | 非学习参照，`optimality_proven=false`，报告称"顺序分段 A\*" |
| Proposed（IGA+TD3，course-final） | **待 Win** | 课程主行 |
| Distance-TSP+TD3 | 229.0 | Mac，course-final |
| DRL-SA | 229.0 | Mac，course-final，与上行确定性相同 |
| MapGuide 规划器 | 273.0 | 弱规划参照 |
| DDPG / D3QN | 失败 0/120 | 多配方多机交叉验证 |

## 7. 待办总览

- [ ] Win：课程版 Proposed low 跑完并随结果分支推送
- [ ] Win：旧机结果分支（含 formal-v5 三档与高档产物）
- [ ] 双方：DDPG 课程口径定稿
- [ ] Mac：EEGA-TD3 (GA proxy)（P3，等指令）
- [ ] 合并：Win 侧将 feat/mac 的代码提交 merge 回 master
