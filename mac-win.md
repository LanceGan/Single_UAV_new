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

### 课程线（low，归零配方）✅ 盲版已完成（2026-11-01 夜）

**2026-11-01 方法论调整与结果**：顺序基线轨迹栈去除 A2G 可见性——状态去 A2G SNR 维（8D→7D，`--a2g_blind`）、教师 G2A-only MapGuide；环境物理、目标与奖励不变。

| 方法 | 顺序 | 成功率 | eval_best T+τ | 目录 |
|---|---|---|---|---|
| Distance-TSP+TD3（**A2G-blind**） | `[1,6,5,2,3,4]` | 56/120 (47%) | **311.0s**（ep1450，T=185.5，τ=125.5，T_out=20.5） | `journal_results/course-baselines/distance-tsp/.../td3_bc/` |
| DRL-SA（**A2G-blind**） | 同上 | 56/120 | **311.0s**（确定性复制，逐位相同） | `journal_results/course-baselines/drl-sa/.../td3_bc/` |
| （消融存档）同基线 A2G 可见版 | 同上 | 91/120 | 229.0s | `journal_results/course-baselines-a2g-visible/` |

**A2G 价值的量化证据**：同顺序、同配方、仅去掉轨迹层 A2G 可见性 → T+τ 恶化 **+82.0s（+35.8%）**、成功率 −29pp、τ 从 89.0→125.5（+41%）。配合 Proposed（A2G 全知）对比，论文可把 A2G 贡献拆成两层呈现。

**顺序质量核查**（`journal_results/analysis-astar-order-check/`）：A\* 下 IGA 序 214.0 < 基线序 216.5（仅低档；Win 侧 §3 发现中高档 TSP 序反而略优，叙事以 Win 版为准）。

### ⚠️ 晨间待裁决（双机立场冲突，2026-11-01 夜记录）

| # | 议题 | Win 立场（win-mac.md / 5adb624） | Mac 已执行（本会话用户指示） |
|---|---|---|---|
| 1 | 谁盲 A2G | 仅 DRL 基线（DDPG/DQN）盲化；序列基线轨迹栈保持全知 | 序列基线轨迹栈全盲（上表 311.0 即其结果）；DDPG/D3QN 保持全知 |
| 2 | 教师下限 | 0.2（与论文 TD3 行配方一致） | 基线按课程文档用归零（可见版 229.0 与盲版 311.0 均为归零） |
| 3 | 奖励版本 | Win 已切 v5（0.5 s/MB） | Mac 保持旧奖励（与论文 TD3 行可比）；基线全部旧奖励 |

无论 #1 如何裁决，两套基线数据（盲 311.0 / 可见 229.0）都已在库：一套做正式基线、另一套做 A2G 消融，不浪费。

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

## 7. 待办总览（2026-11-02 大清零）

- [x] 层面 1/2/3 低档全部实验完成（见 §8/§9 与 win-mac §9）
- [ ] **Win：推送旧机 formal-v5 产物分支（唯一剩余事项，清单见 §10）**
- [x] 口径定稿（2026-11-02 用户拍板，不再讨论）：
  - 教师下限差异（TD3 行 0.2 / 盲基线归零 / D3QN 常数 0.5）：**接受，仅内部知晓，论文不注明**；
  - 多种子：**不加**，单种子可行结果即为论文结果；
  - 中/高档扩展：**不扩**，基线/QoS/权重维持低档；
  - 奖励版本：**旧奖励为论文正式口径**（进论文的全部数字均为旧奖励；v5 仅存在于已作废的 d3qn-v3/v4 尝试，Win 侧请将 v5 改动隔离在自己分支，勿再用于论文 run）。
- 合并 master 的时机仍由 Win 定（master 冻结中）。

## 10. 旧机 formal-v5 产物推送清单（Win 执行）

期刊线 TD3 高档（501.0s）目前只有交接文档里的数字，缺原始产物，导致 Mac 无法生成期刊三档对比图与高档轨迹图。请在**原电脑**上执行：

```bash
# 在原电脑的仓库里（确认 journal_results/formal-v5-restart 存在）
git checkout -b feat/win-legacy
# .gitignore 需放行结果但排除大文件（参照 feat/mac 的做法）：
#   journal_results/**/models/
#   journal_results/**/logs/
git add journal_results/formal-v5-restart
git commit -m "results: formal-v5-restart 三档原始产物（期刊线 TD3 结果）"
git push new feat/win-legacy
```

**必须包含**（每档 td3_bc 目录下）：`run_config.json`、`eval_history.json`（120 次评估全记录）、`UAV_TD3_improved.npz`、`evaluation/` 或轨迹 npz（若有）、`paper_figure_summaries.json`。
**必须排除**：`models/`（checkpoint，数百 MB）、`logs/`（tfevents，单文件可超 GitHub 100MB 硬限）。
**注意**：高档（high/td3_bc）是刚需；low/medium 一并推上便于交叉核对。推送后 Mac 自动接手：拉取 → 生成期刊三档对比图（T/τ/T_out vs 负载）+ 高档 eval_best 双地图轨迹图 → 归档 feat/mac。

## 8. 三组实验设计（2026-11-01 用户定稿）与 Mac 进行中事项

**组 1（全知轨迹算法对比）**：固定 IGA 序 + 双图全知，TD3 vs D3QN vs A\*。TD3 235.5 ✅、A\* 214.0 ✅、D3QN 等 Win 重跑。⚠️ 奖励版本需同组统一（TD3 旧奖励 / D3QN v5）。

**组 2（全知 Proposed 场景敏感性）**：固定 IGA+TD3+全知，扫数据量与 QoS。数据量轴 ✅（235.5/351.5/501.0）。**QoS 轴 ✅ 五点完整（2026-11-02）**：低档 `T_out^max ∈ {30, 40, 60, 80, 100}`——

| QoS 预算 | eval_best T+τ | 实际 T_out | 成功率 |
|---|---|---|---|
| 30s | 248.5（ep1925） | **16.0s** | 72/120 (60%) |
| 40s | 240.5（ep2200） | 21.0s | 88/120 (73%) |
| 60s | 229.0 | 18.5s | 95/120 (79%) |
| 80s | 229.0 | 20.0s | 81/120 (68%) |
| 100s | 235.5 | 21.5s | 97/120 (81%) |

**解读**：策略固有中断需求 ~20s；预算 ≥60s 时曲线平坦（229–235.5，差异在噪声内），**预算逼近 20s 时约束开始咬合**——30s 点目标值退化 +19.5s 且策略把 T_out 主动压到 16.0s（花目标值买裕量），成功率降至 60%。A\* 参照五点全 214.0（其最优解在 30s 预算内也可行）。完整曲线图 `journal_results/qos-sweep/qos_sensitivity.{png,pdf,json}`。单种子每点，趋势与机制自洽但勿过度解读单点差异。

**组 3（他文基线，全 A2G-free）**：分工（2026-11-01 定）——**Win 做两个 D3QN 系（DRL-SA=SA(服务时间)+盲D3QN、SA-D3QN=SA(距离)+盲D3QN），Mac 做两个 TD3 系**。吞吐量代理统一用 **G2A 互易估计**（层面 3 不读 A2G 图；原文用各自信道模型，互易估计最贴近该设定，报告注明适配）。

- Distance-TSP+TD3 ✅ 311.0（盲，56/120）；
- **EEGA-TD3 ✅ 279.5（盲，92/120，ep 不详见 eval_history；T=174.0，τ=105.5，T_out=25.5）**，顺序 [1,6,5,3,2,4]；
- **Yang-style+TD3 ✅ 254.0（盲，102/120；T=157.5，τ=96.5，T_out=31.5）**，顺序 [1,6,5,4,2,3]；
- 早先的 SA+TD3 盲版 run（311.0，`course-baselines/drl-sa/`）按新规范**不再是正式 DRL-SA**（原文轨迹为 D3QN）——改标签为 **"G2A-SA+TD3"** 留作会议延续基线/消融，数据不弃；
- GA-proxy 退役：已有忠实 EEGA 边权实现，`ga_proxy` 仅留档。

**层面 3 低档全景（盲基线 vs 全知 Proposed，同配方同奖励）**：A\* 214.0 < **Proposed TD3 235.5** < Yang 254.0 < EEGA 279.5 < Distance-TSP/G2A-SA 311.0。两个叙事点：①所有盲基线劣于全知 Proposed（+7.9% 到 +32%），A2G 感知的价值成立；②盲基线内部 311→254 的巨大差距表明**顺序的数据/吞吐感知度**（Yang>EEGA>纯距离）在无 A2G 时更关键——从独立角度佐证了数据感知排序的论文主张。对比图：`journal_results/comparison-low-layer3/`。

**组 2 权重轴 ✅ 完成（2026-11-02）**：目标改为 w₁T+w₂τ（(0.5,0.5) 与旧奖励逐位一致，现有结果即均衡组；加权 run 按各自目标选 eval_best；shaping 不动）。三组单调权衡成立：

| (w₁, w₂) | T | τ | T+τ | T_out | 成功率 |
|---|---|---|---|---|---|
| (0.8, 0.2) 重T | **140.5** | 85.0 | 225.5 | 27.5 | 103/120 |
| (0.5, 0.5) 均衡 | 153.0 | 82.5 | 235.5 | 21.5 | 97/120 |
| (0.2, 0.8) 重τ | 156.0 | **75.5** | 231.5 | 22.0 | 102/120 |

T 随 w₁ 单调降、τ 反向单调，重 T 组以 +6s 中断换 12.5s 任务时间——权重确实改变策略行为（轨迹分化叠图 `journal_results/weight-sweep/weight_trajectories_dual_maps.*`、T-τ 权衡散点 `weight_tradeoff.*`）。注意 T/τ 耦合使前沿较窄，论文表述为"权重调节侧重点"而非大幅分化。新 run 在 `journal_results/weight-sweep/`，旧结果零改动。

**顺序差异化全景**（低档）：IGA `[1,6,2,3,5,4]` / TSP `[1,6,5,2,3,4]` / EEGA `[1,6,5,3,2,4]` / Yang `[1,6,5,4,2,3]`——四种方法四种顺序，序列对比叙事成立。

## 9. D3QN 修复版规范（Win 执行，代码在 feat/mac @ `b8ca85a`）

针对 win-mac §9 的失败诊断（长时域信用分配 + 平滑绕行），feat/mac 已实现三件套修复，**Mac 不跑，由 Win 执行**（Win 空闲且有 CUDA）：

### 9.1 修复内容（均为已发表机制，非调参偏方）

1. **n-step 回报（n=10）**：回放存双时距字段，TD 损失 = 1步 + n步 + DQfD margin。**DRL-SA 原文本就使用 Nₘ-step，此前复现缺失此机制**——这是补齐忠实度。回合末截断由 done 掩蔽 bootstrap，固定 γᵏ 精确。
2. **动作重复（k=2）**：每决策执行 2 时隙（50m 粒度），决策地平线 ~350→~175（Atari frame-skip 同款）。评估仍逐时隙 argmax（更细，不吃亏）。
3. **DQfD 预训练**：先收集 40 个纯示范回合 → 2 万步 demo-only 更新（margin+双时距 TD）→ 再在线交互，ε 从零起算。

默认参数向后兼容（n=1/k=1/pretrain=0 ≡ 旧行为，测试锁定；全套 185 测试通过，真实冒烟已验证）。

### 9.2 ⚠️ 代码获取方式：独立 checkout，不要 merge

feat/mac 与 feat/win 代码已分叉（Mac=旧奖励 + `a2g_blind` 参数；Win=v5 奖励 + `include_a2g_snr`），直接 merge 会冲突。推荐：

```bash
git worktree add ../Single_UAV_mac feat/mac   # 或新 clone 后 checkout feat/mac
cd ../Single_UAV_mac
```

**奖励版本有意用旧奖励**：组 1 的对比锚点是 TD3 235.5（旧奖励），D3QN 修复版同口径才公平。若 Win 坚持 v5，需把三件套移植到自己分支并接受与锚点的奖励差异（不推荐）。

### 9.3 启动命令（低档，组 1 口径：A2G-aware + 常数 0.5 引导）

```bash
python Train_D3QN.py --data_level low --teacher mapguide \
  --guide_prob 0.5 --guide_min_prob 0.5 \
  --learning_rate 1e-4 --episodes 5000 \
  --n_step 10 --action_repeat 2 \
  --pretrain_demo_episodes 40 --pretrain_updates 20000 \
  --epsilon_decay_steps 300000 \
  --seed 1 --output_root journal_results/d3qn-v5-fixed
```

CUDA 自动启用（--device 缺省自检）。预计 5000 回合在 CUDA 上 2–3 小时。

### 9.4 判定结果（2026-11-02 已出，走分支一）✅

修复版成功——见 win-mac.md §9（Win 执行并记录）：层面 1 A2G-aware **274.5s**（末千回合成功率 67.5%）；层面 3 盲版 SA-D3QN **289.0s**、DRL-SA **296.5s**。D3QN 结论反转为"补齐原文 n-step 等机制后离散可学，但绕行粗、劣于连续 TD3 约 14%"。层面 3 完整排序（全盲）：Yang 254.0 < EEGA 279.5 < SA-D3QN 289.0 < DRL-SA 296.5 < TSP+TD3 311.0，全部劣于全知 Proposed 235.5。**全三层低档实验至此全部完成。**
