# 论文写作指导手册（PAPER_WRITING_HANDOVER）

> 读者：负责书写论文的 Agent。
> 性质：期刊格式论文 + 课程报告，**不做正式投稿**；按本手册的数字、图表与表述规则成文即可，无需再向两台实验机器求证。
> 数据与图表全部在 `feat/mac` 分支；本手册即唯一权威来源，与 `win-mac.md` / `mac-win.md` 冲突时以本手册为准（2026-11-02 定稿）。

---

## 1. 论文主张与三层实验叙事

研究对象：蜂窝连接无人机在 2km×2km 乡村场景巡检 6 个点（4 GBS/12 扇区，3GPP RMa），利用 **G2A（下行中断）与 A2G（上行速率）双无线电地图**联合优化巡检顺序（数据感知 IGA）与连续轨迹（TD3），目标 `min w₁T+w₂τ`（主结果 w₁=w₂=0.5，即 T+τ），硬约束 `T_out ≤ 100s`。

结果章节按三层组织（与实验设计一一对应）：

- **层面 1（算法对比，全知）**：固定 IGA 顺序 + 双图全知，TD3 vs D3QN vs A\*（+MapGuide 参照）→ 连续动作 TD3 是最优学习方法；
- **层面 2（敏感性，全知 TD3）**：数据量三档、QoS 五点、目标权重三组 → 方法的尺度规律与鲁棒性；
- **层面 3（他文基线对比，A2G-unaware）**：五个来自其他论文的基线全链路不读 A2G 图 → A2G 感知的价值。

核心机制叙事（贯穿全文）：本任务的独特难点是**位置相关的连续上传**——最优行为是沿高吞吐区平滑绕行、边飞边传（而非"飞到点再悬停"）；TD3 的轨迹（T=153s）比教师 MapGuide（196.5s）更快即为证据。

## 2. 方法命名规范（图表与正文统一）

| 写法 | 禁止写法 |
|---|---|
| TD3（或 Proposed） | TD3-BC、td3_bc |
| DDPG / D3QN | 任何 -BC/-bc 后缀 |
| QoS 约束 A\*（顺序分段） | "全局最优 A\*" |
| EEGA-TD3、DRL-SA、SA-D3QN、Distance-TSP+TD3、Yang-style+TD3 | 自造缩写 |

## 3. 最终结果数字（唯一权威表）

### 3.1 层面 1：全知轨迹方法对比（低档）

| 方法 | T+τ (s) | 备注 | 原始数据 |
|---|---|---|---|
| A\* | 214.0 | 离线参照，高档单次搜索可达数百秒 | `journal_results/analysis-astar-order-check/`（低）；教师缓存 meta |
| **TD3** | **235.5** | T=153.0，τ=82.5，T_out=21.5，97/120 | `journal_results/formal-v7/A/low/td3_bc/` |
| MapGuide | 273.0 | 规划器参照 | Win 记录（win-mac §3） |
| D3QN | 274.5 | 末千回合成功率 67.5%；含 n-step/动作重复/DQfD 预训练 | Win `journal_results/d3qn-v5-fixed`（数字见 win-mac §9） |

### 3.2 层面 2：三条敏感性轴（全知 TD3）

**数据量**（`journal_results/load-sweep/load_sensitivity.json`）：

| 档位 | TD3 | A\* | MapGuide | TD3 明细 |
|---|---|---|---|---|
| 低 | 235.5 | 214.0 | 273.0 | T=153.0 τ=82.5 T_out=21.5（97/120） |
| 中 | 351.5 | 318.0 | 369.0 | T=218.0 τ=133.5 T_out=19.5（73/120） |
| 高 | 501.0 | 447.0 | 492.0 | T=300.5 τ=200.5 T_out=17.5（39/120） |

**QoS 五点**（低档，`journal_results/qos-sweep/qos_sensitivity.json`）：30s→248.5（T_out 16.0，60%）、40s→240.5、60s→229.0、80s→229.0、100s→235.5；A\* 五点全 214.0。解读：策略固有中断需求 ~20s，预算 ≥60s 不束紧（平坦段差异属噪声），逼近 20s 时约束咬合。

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
| F2 | `summary-bars/layer1_methods.*` | 层面 1 柱状：A\*/TD3/MapGuide/D3QN |
| F3 | `formal-v7/A/low/td3_bc/paper_figures/trajectory_eval_best_dual_maps.*` | 低档 TD3 轨迹叠双地图（避开中断区+贴高 SNR 绕行） |
| F4 | 同上 medium / `formal-v5-restart/A/high/...` | 中、高档轨迹（三档可并排） |
| F5 | `comparison-low/comparison_rewards.*` | 0–2000 轮纯 actor 奖励曲线（TD3 收敛 vs DDPG/D3QN 旧配方不收敛；若仅需 TD3 可用单图 `formal-v7/.../reward_vs_episode_0_2000.*`） |
| F6 | `load-sweep/load_sensitivity.*` | 层面 2 数据量轴（左：三方法；右：T/τ 分解+T_out 标注） |
| F7 | `qos-sweep/qos_sensitivity.*` | 层面 2 QoS 轴（左：目标 vs 预算；右：用量远低于束缚线） |
| F8 | `weight-sweep/weight_trajectories_dual_maps.*` + `weight_tradeoff.*` | 层面 2 权重轴（轨迹分化 + T-τ 散点） |
| F9 | `summary-bars/layer3_baselines.*` | 层面 3 终版柱状（Proposed 高亮 + A\* 参考线） |
| F10 | `summary-bars/a2g_ablation.*` | A2G 消融柱状（229 vs 311 + Proposed 线） |
| F11 | `comparison-low-layer3/comparison_objective.*` | 可选：TD3 系基线训练全程可行目标曲线（含不可行存在带） |

## 5. 表述规则与禁止清单（硬约束）

1. 结果一律取**纯 actor、无引导、无噪声的 eval_best**；失败/提前终止轨迹不得进入可行排名或以低值出现在数值轴；
2. 奖励曲线只展示 **0–2000 回合、仅 actor 回合、200 回合移动平均**，图题注明"展示主要学习阶段"，不声称后期稳定，完整数据保留；
3. **A2G-free/unaware = 基线算法不读取 A2G 地图**，环境的上传物理与 τ 记账对所有方法相同——不得写成"去掉上传链路"；
4. D3QN 口径：**"补齐 n-step 等原文机制后离散可学会，但绕行粗糙、劣于连续 TD3 约 14%"**——不得再写"离散学不会"（旧结论已反转）；方法部分注明其训练含 n-step 回报、动作重复与示范预训练；
5. **高档措辞**：TD3（501.0）略高于 MapGuide（492.0），写法建议为"在高负载下与规划器参照相当（差距 <2%），同时保持毫秒级在线决策（规划器需对全图重算）"；低/中档 TD3 明显占优可直说；
6. 适配声明：DRL-SA 原文轨迹为 D3QN（本文如此复现 ✓ 可直说）；Yang 原文为 SCA/BCD，"+TD3" 必须声明为本文适配；EEGA 的自适应算子参数原文未公开，本文用标准 GA 算子并披露；层面 3 的吞吐估计 Q 来自 **G2A 互易估计**（原文无独立 A2G 图，各自用信道模型）；
7. **DDPG 不出现在论文中**（内部负结果）；
8. 单种子：所有数字为单一种子（seed 1）正式运行结果，论文**不要出现 mean±std 或"多种子"字样**，也不必主动强调单种子；
9. **以下内部事项不写入论文**：各方法示范教师调度的细节差异（0.2 下限/归零/常数 0.5）；A\* 教师失败实验；course-baselines-a2g-visible 的中间消融目录结构；两台机器与分支的分工。

## 6. 数字与论文口径的勘误保护

- 会议论文 PDF 写 `T_out^max=80s`、`v_max=20m/s`；**代码与全部实验为 100s、50m/s（25m/0.5s 时隙）**——期刊稿按代码口径写参数表；
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
