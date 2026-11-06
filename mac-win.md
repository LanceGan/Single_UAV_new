# Mac–Win 双机同步备忘

> 维护：Mac 侧（每次 Mac 有新结果/新约定时更新本文件并推送 `feat/mac`）
> 最后更新：2026-11-01
> 目的：两台机器并行做实验，用此文件对齐"谁在跑什么、哪份结果算数、代码在哪个分支"，避免重复劳动和口径混乱。

> **2026-11-05 约束变更**：正式 QoS 硬约束已改为“最长一次连续中断”`T_out^cont`。本文中未特别说明的旧 `T_out` 数字均为整任务累计中断口径，只作追溯；新版正式实验按 `QOS_RETRAINING_PLAN.md` 重做。

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
|---|---|---|---|---|---|
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
| PSO-trajectory | 249.5 | 离线元启发式轨迹（Mac，2026-11-02 新增）：T=136.5 最快但 τ=113.0/T_out=46.0，全知+单实例 25 分钟求解仍劣于 TD3 ——回答"为何不用元启发式求轨迹" |
| MapGuide 规划器 | 273.0 | 弱规划参照 |
| DDPG / D3QN | 失败 0/120 | 多配方多机交叉验证 |

## 7. 待办总览（2026-11-02 大清零）

- [x] 层面 1/2/3 低档全部实验完成（见 §8/§9 与 win-mac §9）
- [x] **Win：旧机 formal-v5 产物分支已推送（feat/win-legacy）并由 Mac 接手完成**（2026-11-02）：三档数字逐位核验一致（244.0/353.5/501.0，DDPG 0/120）；期刊三档对比图已生成 `journal_results/load-sweep/load_sensitivity.{png,pdf,json}`（左：三方法目标 vs 负载；右：TD3 的 T/τ 分解 + T_out 标注）；高档 eval_best 轨迹图按论文命名（"TD3"）从轨迹 npz 重渲染。**注意叙事点**：高档 TD3（501.0）略差于 MapGuide（492.0，-1.8%），低/中档 TD3 明显占优——论文高档措辞需照此处理。
- [x] 口径定稿（2026-11-02 用户拍板，不再讨论）：
  - 教师下限差异（TD3 行 0.2 / 盲基线归零 / D3QN 常数 0.5）：**接受，仅内部知晓，论文不注明**；
  - 多种子：**不加**，单种子可行结果即为论文结果；
  - 中/高档扩展：**不扩**，基线/QoS/权重维持低档；
  - 奖励版本：**旧奖励为论文正式口径**（进论文的全部数字均为旧奖励；v5 仅存在于已作废的 d3qn-v3/v4 尝试，Win 侧请将 v5 改动隔离在自己分支，勿再用于论文 run）。
- 合并 master 的时机仍由 Win 定（master 冻结中）。

## 8. 三组实验设计（2026-11-01 用户定稿）与 Mac 进行中事项

**组 1（全知轨迹算法对比）**：固定 IGA 序 + 双图全知，TD3 vs D3QN vs A\*。TD3 235.5 ✅、A\* 214.0 ✅、D3QN 等 Win 重跑。⚠️ 奖励版本需同组统一（TD3 旧奖励 / D3QN v5）。

**组 2（全知 Proposed 场景敏感性）**：固定 IGA+TD3+全知，扫数据量与 QoS。数据量轴的原累计口径结果暂留档。**2026-11-05 用户裁决：正式 QoS 硬约束改为最长一次连续中断 `T_out^cont`。** 现有主要轨迹的 `T_out^cont` 约为 2.5–4.5s；MapGuide 预检在 3/4/6s 下均为 2.5s。新版 A* 顺序分段标定：1–3s 无可行路径，4s 为 219.0s，5–6s 为 216.0s。

| 正式预算 | 3s | 4s | 5s | 6s | 8s |
|---|---|---|---|---|---|
| 实验状态 | 待重训 | 待重训 | 待重训 | 待重训 | 待重训 |

旧 `qos-sweep/` 与 `qos-sweep-reweighted/` 都是累计中断口径历史产物，只保留追溯，不进入新版论文。

**正式 QoS 实验交 Win**：按照 `QOS_RETRAINING_PLAN.md` 对 3 个权重 × `{3,4,5,6,8}s` 共 15 组分别从头训练。新旧约束的结果不得混合。

**连续 DRL baseline 变更（2026-11-05）**：DDPG 旧结果的两种 MapGuide 配方均为 0/120 成功，保留为内部负例；新版层面 1 增加 PPO 连续动作基线，训练入口为 `Train_PPO.py`，完整命令见 `RERUN_ASSIGNMENT.md`。PPO 与 TD3 使用相同状态、动作界、IGA 顺序、奖励和连续中断约束。

**2026-11-05 全量重跑分工**：Win 负责 15 组 QoS、PPO 和 Proposed 中高档；Mac 负责层面 1 D3QN、三个 TD3 基线、A2G 消融、DRL-SA/SA-D3QN、A*、MapGuide、PSO 和顺序审计。6 s 是主方法和 baseline 的统一比较预算；6 s 三权重结果同时复用为 Proposed 低档和权重实验。具体矩阵和复用关系见 `RERUN_ASSIGNMENT.md`。Win 今天用累计中断口径跑出的 QoS 结果全部作废留档。

**组 3（他文基线，全 A2G-free）**：新版分工——**Mac 整体执行三个 TD3 系和两个 D3QN 系**（Distance-TSP、EEGA、Yang、DRL-SA、SA-D3QN），避免跨机拆分同一基线家族。吞吐量代理统一用 **G2A 互易估计**（层面 3 不读 A2G 图；原文用各自信道模型，互易估计最贴近该设定，报告注明适配）。

`Train_D3QN.py` 的 v4 入口现支持 `--sequence_method {iga,distance_tsp,sa,sa_service,sa_distance}` 和 `--a2g_blind`。正式命令：DRL-SA 用 `sa_service + a2g_blind`，SA-D3QN 用 `sa_distance + a2g_blind`；两者均为 7 维 G2A-only 状态，完整命令见 `RERUN_ASSIGNMENT.md`。

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

## 10. 旧机 formal-v5 产物推送清单（Win 执行）✅ 已完成（feat/win-legacy，Mac 已接手出图）

## 11. d3qn-v5-fixed 数据推送清单（Win 执行，论文奖励曲线第四条线急需）

论文的奖励曲线图要加入修复版 D3QN 的收敛曲线，写作侧缺曲线数据。请在 Win 上把修复版 D3QN run 的以下文件推上来（追加到 `feat/win-legacy` 分支即可）：

```bash
# 在跑 d3qn-v5-fixed 的那个 checkout（应为 feat/mac 的 worktree）里：
git checkout feat/win-legacy   # 或在原仓库 cherry 对应文件后推送
git add journal_results/d3qn-v5-fixed/low/d3qn/training_history.npz \
        journal_results/d3qn-v5-fixed/low/d3qn/run_config.json \
        journal_results/d3qn-v5-fixed/low/d3qn/eval_history.json
git commit -m "results: d3qn-v5-fixed 低档曲线数据与配置留档"
git push new feat/win-legacy
```

**必须包含**：`training_history.npz`（含 `episode_rewards` 与 `episode_guided`，奖励曲线靠它）；**建议一并**：`run_config.json`、`eval_history.json`（留档审计）。**排除**：`models/`、`logs/`（老约定）。
**顺带可选（避免下次再跑一趟）**：`drl-sa`、`sa-d3qn` 两个修复版 run 的同样三件文件。

⚠️ **推送时请确认并在提交信息里写明奖励版本**：目录名 `d3qn-v5-fixed` 的 "v5" 是**第五次尝试的编号，不是奖励 v5**。按 §9.3 规范该 run 应从 feat/mac worktree 启动 = **旧奖励**（与 TD3/DDPG 曲线同尺度，图注无需提奖励版本）。若实际是在 Win 自己的 v5 奖励分支上跑的，必须说明——那样四条曲线奖励尺度不同，图注必须注明不可比。核验方法：该 checkout 的 `World.py` 若含 `UPLOAD_LEG_REWARD = 10.0` 且无 0.5/MB 平率逻辑即为旧奖励。

推送完成后：Mac/写作侧 `git fetch` 取数，重跑奖励曲线图（加第四条线），配文补一句 D3QN 收敛描述。

## 12. Mac 端论文写作交接（2026-11-04）

Mac 端的正式实验已经按论文锁定范围全部完成，后续论文写作统一交给 Win 端。两套仓库的分工如下：

| 仓库 | 远程 | 交接分支 | 内容 |
|---|---|---|---|
| 实验仓库 `Single_UAV_new` | `git@github.com:LanceGan/Single_UAV_new.git` | `feat/mac` | 代码、正式结果、图表生成工具、双机备忘、论文写作手册 |
| 论文仓库 `P3` | `git@github.com:LanceGan/P3_journal.git` | `feat/mac` | `journal.tex`、`refs.bib`、`figs/` 和当前编译版 `P3.pdf` |

Win 端写作时应以：

1. `PAPER_WRITING_HANDOVER.md` 作为数字、图注和方法表述的唯一权威口径；
2. 实验仓库的 `journal_results/` 核对原始评估与图；
3. 论文仓库的 `feat/mac` 作为当前 Mac 端稿件基线。

Mac 端论文仓库已经成稿，不再把“待论文成稿”列为实验待办。当前收尾项只有目标期刊适配和终稿检查。注意 `P3.pdf` 的最后编译时间早于最近一次标题修改，正式审阅前需要重新编译。

训练 checkpoint（`models/`）与原始 TensorBoard 日志（`logs/`）继续按既有约定保留在本机，不进入 GitHub；论文写作和结果核对所需的配置、评估历史、轨迹、训练曲线及图表均已同步。

## 13. v4 图表同步与剩余轨迹模型（2026-11-06）

### 13.1 已完成并同步的定量图

论文仓库 `figs/v4/` 已生成并同步以下图：

- `qos_sensitivity_v4`
- `objective_weight_sensitivity_v4`
- `load_sensitivity_v4`
- `layer1_methods_v4`
- `layer3_baselines_v4`
- `a2g_ablation_v4`
- `training_returns_v4`

每张图提供 PDF、SVG、600 dpi PNG；源数据见
`figs/v4/source_data/figure_source_data_v4.{csv,json}`。图注、数字口径和旧版
正文中必须删除的表述见 `figs/v4/README_FOR_WIN.md`。

所有定量图均通过：

- 多面板最终几何对齐：PASS（两张单面板图按规范为 NOT APPLICABLE）
- PDF 字形下限：最低 6.5 pt
- PDF 渲染碰撞审计：7/7 PASS
- 绘图源码严格预检：21/21 PASS

### 13.2 Win 需要补传的轨迹模型

Win 已经推送的 `eval_history.json`、`run_config.json`、
`training_history.npz`/`UAV_TD3_improved.npz` 和预检轨迹足以重画全部二维
定量图，但不足以重画论文中的 **eval-best 双地图轨迹**。预检轨迹来自
MapGuide，不是最终学习策略，不能替代 `eval_best` 轨迹。

请在 Win 的 v4 训练 checkout 中收集以下 5 个 Proposed actor 检查点：

| 目标标签 | 源文件（相对实验仓库根目录） |
|---|---|
| 权重 `(0.8, 0.2)`，6 s | `journal_results/v4-qos/w80-20/qos6/A/low/td3_bc/models/td3_bc_td3_actoreval_best.pth` |
| 权重 `(0.5, 0.5)`，6 s | `journal_results/v4-qos/w50-50/qos6/A/low/td3_bc/models/td3_bc_td3_actoreval_best.pth` |
| 权重 `(0.2, 0.8)`，6 s | `journal_results/v4-qos/w20-80/qos6/A/low/td3_bc/models/td3_bc_td3_actoreval_best.pth` |
| Medium，6 s | `journal_results/v4-models/load/medium/A/medium/td3_bc/models/td3_bc_td3_actoreval_best.pth` |
| High，6 s | `journal_results/v4-models/load/high/A/high/td3_bc/models/td3_bc_td3_actoreval_best.pth` |

`journal_results/**/models/` 被 `.gitignore` 排除，不能直接 `git add` 原路径。
请在 Win 仓库中新建不受忽略的同步目录，并按以下稳定名称复制：

```text
sync_bundles/v4_eval_best_actors/
  w80-20_qos6_actor.pth
  w50-50_qos6_actor.pth
  w20-80_qos6_actor.pth
  medium_qos6_actor.pth
  high_qos6_actor.pth
  MANIFEST.json
```

`MANIFEST.json` 至少记录每项的：

```json
{
  "label": "w50-50_qos6",
  "source_run": "journal_results/v4-qos/w50-50/qos6/A/low/td3_bc",
  "eval_best_episode": 1550,
  "sha256": "..."
}
```

不要传 critic、历史 period checkpoint 或 TensorBoard 日志；轨迹评估只需要
actor。5 个文件预计合计约 3 MB。打包后提交并推送 `feat/win`，同时在
`sync_bundles/v4_eval_best_actors/README.md` 写明训练 checkout 的 commit。

可选的 PPO 失败轨迹需要额外同步：

```text
journal_results/v4-models/ppo/A/low/ppo/models/ppo_ppo_actoreval_best.pth
```

当前七张定量图不依赖该 PPO checkpoint。

### 13.3 Mac 侧尚待补齐的数据

为了在负载敏感性图中加入同预算的 A* 离线参照，Mac 还需要在 6 s 下补齐：

- Low：`journal_results/v4-deterministic/astar-main-6/low/`
- Medium：`journal_results/v4-deterministic/astar-main-6/medium/`

现有 `astar-main-6` 只有 High 的成功重放结果。`astar-main-8` 的低档不是成功
重放，不能拿来做 6 s 对照。

### 13.4 收到模型后的验收

Mac 收到 actor bundle 后将：

1. 校验 SHA-256 与 `eval_best.episode` 和现有 `eval_history.json` 一致；
2. 用 `evaluate_policy.py` 在 CPU 上无噪声、无 MapGuide 重放；
3. 断言 `T`、`tau`、`T_out^cont`、巡检点和上传量与该 run 的
   `eval_history.json` 最佳记录一致；
4. 重画 Proposed 三档、三权重和 A2G 基线的双地图轨迹图；
5. 运行字号、对齐和 PDF 碰撞审计后再同步到论文仓库。
