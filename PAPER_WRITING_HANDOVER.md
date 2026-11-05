# 论文写作指导手册（PAPER_WRITING_HANDOVER）

> 读者：负责书写论文的 Agent。
> 性质：期刊格式论文 + 课程报告，**不做正式投稿**；按本手册的数字、图表与表述规则成文即可，无需再向两台实验机器求证。
> 数据与图表全部在 `feat/mac` 分支；本手册即唯一权威来源，与 `win-mac.md` / `mac-win.md` 冲突时以本手册为准。
>
> **2026-11-05 重大口径变更**：正式 QoS 硬约束已由“整任务累计中断”改为“最长一次连续中断”。本手册中所有基于旧累计约束的实验数字均为 legacy，只能用于追溯，不能直接写入新版论文。新版正式结果必须由 `QOS_RETRAINING_PLAN.md` 的重训实验生成。

---

## 1. 论文主张与三层实验叙事

研究对象：蜂窝连接无人机在 2km×2km 乡村场景巡检 6 个点（4 GBS/12 扇区，3GPP RMa），利用 **G2A（下行中断）与 A2G（上行速率）双无线电地图**联合优化巡检顺序（数据感知 IGA）与连续轨迹（TD3），目标 `min w₁T+w₂τ`（主结果 w₁=w₂=0.5，即 T+τ），硬约束为 `T_out^cont ≤ T_out^cont,max`。`T_out^cont` 是最长一次连续中断；整任务累计中断 `T_out^total` 仅作辅助诊断。

结果章节按三层组织（与实验设计一一对应）：

- **层面 1（算法对比，全知）**：固定 IGA 顺序 + 双图全知，TD3 vs D3QN vs A\*（+MapGuide 参照）→ 连续动作 TD3 是最优学习方法；
- **层面 2（敏感性，全知 TD3）**：数据量三档、QoS 五点、目标权重三组 → 方法的尺度规律与鲁棒性；
- **层面 3（他文基线对比，A2G-unaware）**：五个来自其他论文的基线全链路不读 A2G 图 → A2G 感知的价值。

核心机制叙事（贯穿全文）：本任务的独特难点是**位置相关的连续上传**——最优行为是沿高吞吐区平滑绕行、边飞边传（而非"飞到点再悬停"）；TD3 的轨迹（T=153s）比教师 MapGuide（196.5s）更快即为证据。

## 2. 方法命名规范（图表与正文统一）

| 写法 | 禁止写法 |
|---|---|
| TD3（或 Proposed） | TD3-BC、td3_bc |
| PPO / DDPG / D3QN | 任何 -BC/-bc 后缀 |
| QoS 约束 A\*（顺序分段） | "全局最优 A\*" |
| EEGA-TD3、DRL-SA、SA-D3QN、Distance-TSP+TD3、Yang-style+TD3 | 自造缩写 |

## 3. 结果数字（版本区分）

> **Legacy 提醒**：本节以下所有性能表均来自旧“累计中断”硬约束，不能作为新版 QoS 结论。新版正式表格必须等待 `QOS_RETRAINING_PLAN.md` 的 15 组结果。

### 3.1 Legacy 层面 1：全知轨迹方法对比（旧累计中断口径）

| 方法 | T+τ (s) | 备注 | 原始数据 |
|---|---|---|---|
| A\* | 214.0 | 离线参照，高档单次搜索可达数百秒 | `journal_results/analysis-astar-order-check/`（低）；教师缓存 meta |
| **TD3** | **235.5** | T=153.0，τ=82.5，T_out=21.5，97/120 | `journal_results/formal-v7/A/low/td3_bc/` |
| PSO-trajectory | 249.5 | 离线元启发式轨迹规划（28 维航路点+悬停编码，全知双地图，单实例求解 ~25 分钟）；T=136.5 最快但 τ=113.0、T_out=46.0——"飞得快、传得差"，双地图精细利用正是其短板 | `journal_results/pso-trajectory/low/` |
| MapGuide | 273.0 | 规划器参照 | Win 记录（win-mac §3） |
| D3QN | 274.5 | 末千回合成功率 67.5%；含 n-step/动作重复/DQfD 预训练 | Win `journal_results/d3qn-v5-fixed`（数字见 win-mac §9） |

### 3.2 层面 2：三条敏感性轴（全知 TD3）

**数据量**（`journal_results/load-sweep/load_sensitivity.json`）：

| 档位 | TD3 | A\* | MapGuide | TD3 明细 |
|---|---|---|---|---|
| 低 | 235.5 | 214.0 | 273.0 | T=153.0 τ=82.5 T_out=21.5（97/120） |
| 中 | 351.5 | 318.0 | 369.0 | T=218.0 τ=133.5 T_out=19.5（73/120） |
| 高 | 501.0 | 447.0 | 492.0 | T=300.5 τ=200.5 T_out=17.5（39/120） |

**新版 QoS 轴（待 Win 重训）**：正式预算为 `{3,4,5,6,8}s`。现有轨迹的最大连续中断约为 2.5–4.5s；MapGuide 预检在 3/4/6s 下均为 `T_out^cont=2.5s`。新版 A* 标定在 1–3s 未找到可行路径，4s 为 219.0s，5–6s 为 216.0s。完整实验矩阵、命令和验收规则见 `QOS_RETRAINING_PLAN.md`。

连续中断的记账规则：逐 0.5s 时隙判定 G2A 中断；中断时当前连续段累加，非中断时立即清零，并记录历史最大值。硬约束只检查历史最大值。整任务累计值可另报为 `T_out^total`，但不得用它做可行性判定。

**顺序结论更新（2026-11-05）**：新约束下低负载 6s 的 A* 重放为距离 TSP `215.5s`、IGA `216.0s`、EEGA `222.5s`、Yang-style `223.5s`。旧稿“IGA 顺序优于 TSP”的结论已不再成立，最终表述必须等待重训后的 DRL 比较，不得沿用旧数值。

**PSO 结果更新（2026-11-05）**：新版 6s 连续中断约束下，PSO 按 `swarm=40, seed=1, iterations=60` 在第 19 次迭代早停，最佳候选 `T_out^cont=6.5s`，任务失败。失败结果必须保留，不能计入可行排名。

**目标权重**（低档，`journal_results/weight-sweep/weight_summary.json`）：

| (w₁,w₂) | T | τ | T_out |
|---|---|---|---|
| (0.8,0.2) | **140.5** | 85.0 | 27.5 |
| (0.5,0.5) | 153.0 | 82.5 | 21.5 |
| (0.2,0.8) | 156.0 | **75.5** | 22.0 |

T 随 w₁ 单调、τ 反向单调；措辞用"权重调节侧重点"（T/τ 耦合使前沿较窄，勿夸大分化幅度）。

### 3.3 层面 3：A2G-unaware 基线（低档，全部劣于 Proposed）

| 方法 | 顺序算法 | 轨迹 | T+τ (s) | 成功率 |
|---|---|---|---|---|
| **Proposed** | 数据感知 IGA | TD3 | **235.5** | 97/120 |
| Yang-style+TD3 | 吞吐/数据加权（精确） | TD3 | 254.0 | 102/120 |
| EEGA-TD3 | L=(I/Q)·E 加权 GA | TD3 | 279.5 | 92/120 |
| SA-D3QN | SA（距离） | D3QN | 289.0 | 93/200 |
| DRL-SA | SA | D3QN | 296.5 | 53/200 |
| Distance-TSP+TD3 | 精确距离 TSP | TD3 | 311.0 | 56/120 |

顺序差异化：IGA `[1,6,2,3,5,4]` / TSP `[1,6,5,2,3,4]` / EEGA `[1,6,5,3,2,4]` / Yang `[1,6,5,4,2,3]`。盲基线内部 311→254 的阶梯 = 顺序的数据/吞吐感知度价值。

### 3.4 消融与佐证

- **A2G 可见性消融**（同一基线 Distance-TSP+TD3，唯一变量 = 轨迹栈能否读 A2G 图）：可见 229.0（91/120）vs 盲 311.0（56/120）→ **+82.0s（+35.8%）**，τ 从 89.0 恶化到 125.5；
- **顺序质量 A\* 核查**（低档）：IGA 序 214.0 < TSP 序 216.5；
- 教师对比：TD3 235.5 快于其教师 MapGuide 273.0（学生超老师，证明非单纯模仿）。

## 4. 图清单（全部 PNG+PDF 双格式）

| # | 路径（`journal_results/` 下，另注明者除外） | 建议用途 / caption 要点 |
|---|---|---|
| F1 | `paper_figures/algorithm_framework/proposed_iga_td3_framework.*`（仓库根） | 方法框架图 |
| F2 | `summary-bars/layer1_methods.*` | 层面 1 柱状：A\*/TD3/PSO-trajectory/MapGuide/D3QN。叙事点：在线毫秒策略 TD3 优于拿全图离线搜 25 分钟的元启发式（249.5），回答"为何不用 GA/PSO 直接求轨迹" |
| F3 | `formal-v7/A/low/td3_bc/paper_figures/trajectory_eval_best_dual_maps.*` | 低档 TD3 轨迹叠双地图（避开中断区+贴高 SNR 绕行） |
| F4 | 同上 medium / `formal-v5-restart/A/high/...` | 中、高档轨迹（三档可并排） |
| F5 | `comparison-low/comparison_rewards.*` | 0–2000 轮纯 actor 奖励曲线（TD3 收敛 vs DDPG/D3QN 旧配方不收敛；若仅需 TD3 可用单图 `formal-v7/.../reward_vs_episode_0_2000.*`） |
| F6 | 待用新约束重生成 | 层面 2 数据量轴（左：三方法；右：T/τ 分解+`T_out^cont` 标注）。旧图使用累计中断，不得混用 |
| F7 | 待生成 | 新版 QoS 轴必须使用 `T_out^cont` 和 `3/4/5/6/8s` 的逐预算重训结果。旧 `qos-sweep/`、`qos-sweep-reweighted/` 均为累计中断历史产物，不进入新版论文 |
| F8 | `weight-sweep/weight_trajectories_dual_maps.*` + `weight_tradeoff.*` | 层面 2 权重轴（轨迹分化 + T-τ 散点） |
| F8b | `weight-sweep/w80-20/.../paper_figures/reward_vs_episode_0_2000_all.*`、`formal-v7/A/low/td3_bc/.../reward_vs_episode_0_2000_all.*`、`weight-sweep/w20-80/.../paper_figures/reward_vs_episode_0_2000_all.*` | 三个权重组的训练收敛曲线（**用户定稿：用 `_all` 版=全部训练回合的真实奖励，含示范回合**；同窗口同规则并排）。三条均为低→高收敛（秒值显示：起点约 −260~−450，峰值 +480~+590）；(0.8,0.2)/(0.2,0.8) 平台稳定（末段 σ≤0.36），(0.5,0.5) 平台至 ~1280 后有已知波动——caption 用既定句式"最终策略由全程周期性无探索评估选出（eval_best）"。图注注明"训练回合含示范回合（概率 0.5→0.2）"。actor-only 版（无 `_all` 后缀）保留作内部核对，不进论文 |
| F9 | `summary-bars/layer3_baselines.*` | 层面 3 终版柱状（Proposed 高亮 + A\* 参考线） |
| F10 | `summary-bars/a2g_ablation.*` | A2G 消融柱状（229 vs 311 + Proposed 线） |
| F11 | `comparison-low-layer3/comparison_objective.*` | 可选：TD3 系基线训练全程可行目标曲线（含不可行存在带） |
| F12 | `pso-trajectory/low/trajectory_dual_maps.*` | 可选：PSO 轨迹叠双地图——与 TD3 轨迹（F3）对照可见其直线穿越中断区的粗糙绕行 |

## 5. 表述规则与禁止清单（硬约束）

1. 结果一律取**纯 actor、无引导、无噪声的 eval_best**；失败/提前终止轨迹不得进入可行排名或以低值出现在数值轴；
2. 奖励曲线只展示 **0–2000 回合、仅 actor 回合、200 回合移动平均**，图题注明"展示主要学习阶段"，不声称后期稳定，完整数据保留；
3. **A2G-free/unaware = 基线算法不读取 A2G 地图**，环境的上传物理与 τ 记账对所有方法相同——不得写成"去掉上传链路"；
4. D3QN 口径：**"补齐 n-step 等原文机制后离散可学会，但绕行粗糙、劣于连续 TD3 约 14%"**——不得再写"离散学不会"（旧结论已反转）；方法部分注明其训练含 n-step 回报、动作重复与示范预训练；
5. **高档措辞**：TD3（501.0）略高于 MapGuide（492.0），写法建议为"在高负载下与规划器参照相当（差距 <2%），同时保持毫秒级在线决策（规划器需对全图重算）"；低/中档 TD3 明显占优可直说；
6. 适配声明：DRL-SA 原文轨迹为 D3QN（本文如此复现 ✓ 可直说）；Yang 原文为 SCA/BCD，"+TD3" 必须声明为本文适配；EEGA 的自适应算子参数原文未公开，本文用标准 GA 算子并披露；层面 3 的吞吐估计 Q 来自 **G2A 互易估计**（原文无独立 A2G 图，各自用信道模型）；

7. 层面 1 的 D3QN 使用 3000 episodes，与 Proposed TD3 同步；其余修复机制为 `guide_prob=guide_min_prob=0.5`、`n_step=10`、`action_repeat=2`、40 回合/20000 update 的 DQfD 预训练、学习率 `1e-4`。层面 3 的 DRL-SA/SA-D3QN 保留 5000 episodes。DRL-SA 使用服务时间 SA（`sequence_selector=sa_service`，G2A 互易吞吐），SA-D3QN 使用纯距离 SA（`sequence_selector=sa_distance`），两者均为 7 维 A2G-blind 状态。
8. **DDPG 不出现在论文中**（内部负结果）；其连续 DRL 对比位置由 PPO 替代。PPO 使用同一状态、动作边界、IGA 顺序、奖励和连续中断约束，最终评估使用策略均值；
9. 单种子：所有数字为单一种子（seed 1）正式运行结果，论文**不要出现 mean±std 或"多种子"字样**，也不必主动强调单种子；
10. **以下内部事项不写入论文**：各方法示范教师调度的细节差异（0.2 下限/归零/常数 0.5）；A\* 教师失败实验；course-baselines-a2g-visible 的中间消融目录结构；两台机器与分支的分工。

### 5.1 奖励曲线形状的解释口径（写作时可能被问到）

曲线不是传统"平滑从低到高"的原因（可写进正文一两句或仅备答辩）：
1. 回报分布**双峰**——成功≈+5（里程碑+完成奖励）、失败≈−10~−15（终止惩罚），均值曲线本质是成功率的阶梯；
2. 曲线版本（**2026-11-02 用户定稿**）：论文奖励曲线一律用 `_all` 版——全部训练回合真实奖励、**按秒值显示**（论文中奖励函数按秒定义：完成 +200、到点 +50、段上传 +10、不可行 −1000、逐秒 −dt；训练实现中的 0.01 critic 缩放因子属实现细节，不写进公式）、**50 回合移动平均**（保留震荡感）。图注注明"训练回合含示范回合（概率 0.5→0.2）"。actor-only 版与 200 窗口版仅作内部核对；
3. **各档位渐近回报不同**（中/高档回合更长、τ 段更长），三档曲线不会收敛到同一水平，高档成功率低则均值长期被失败值压低——分档出图，不要并轴比较绝对值；
4. TD3 的**后期策略退化**（~1300 回合后，v5 与 v7 均有）使曲线非单调——这正是采用周期性无引导评估选 eval_best 的原因（既定句式见 §5 第 2 条）。

## 6. 数字与论文口径的勘误保护

- **A\* 数字的可信度口径**（被质疑时用）：报告值为**真实 World 重放账本**（`metrics.json` 的 `comparison_metrics_source=world_replay`），规划器内部量化目标与重放值逐位一致（214.0==214.0，零乐观偏差）；三次独立计算（教师缓存/Mac 重放/Win 独立运行）互相印证；且因 25m 栅格+逐航段搜索（`optimality_proven=false`），214.0 实为真最优的上界估计——连续最优只会更低；
- 会议论文 PDF 写旧累计预算 `T_out^max=80s`、`v_max=20m/s`；新版代码使用最长连续中断和 `v_max=50m/s`（25m/0.5s 时隙）。新版正式 QoS 预算为 `3/4/5/6/8s`，不得继续写 80s 或 100s；
- 面积 2km×2km、6 巡检点、4 GBS、载频/带宽等以 `run_config.json` 为准（每个 run 目录都有，含地图哈希）；
- 低/中/高负载向量：`journal/data_vectors.json`（[90,90,60,60,90,60]/[180×3,120×3]/[210,290,290,290,210,210] MB）。

## 7. 复现命令速查（审稿/附录用，可选）

```bash
# Proposed（每档一次）
python Train_SINR.py --variant A --data_level {low,medium,high} --algorithm td3_bc --teacher mapguide --seed 1 --episodes 3000
# 层面 3 盲基线（示例：Yang）
python Train_SINR.py --variant A --data_level low --algorithm td3_bc --teacher mapguide --guide_min_prob 0 \
  --sequence_method yang --a2g_blind --seed 1 --episodes 3000
# D3QN（修复配方）
python Train_D3QN.py --data_level low --teacher mapguide --guide_prob 0.5 --guide_min_prob 0.5 \
  --learning_rate 1e-4 --episodes 5000 --n_step 10 --action_repeat 2 \
  --pretrain_demo_episodes 40 --pretrain_updates 20000
# A* / 图表
python run_astar.py --levels low --mode sequential
python plot_load_sweep.py / plot_qos_sweep.py / plot_weight_tradeoff.py / plot_result_bars.py ...
```
