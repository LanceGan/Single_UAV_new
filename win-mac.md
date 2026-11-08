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

- **D3QN 正确配方已基本跑完（Windows，层面 1）**：A2G-aware 8 维 + guide 0.5 常数 + v5 奖励，3000 轮，输出 `journal_results/d3qn-v4`。**0/120 成功（3000 轮跑完，最终判定失败）**，详见 §9。
- **DRL-SA / SA-D3QN 已完成（Windows，层面 3，修复版 D3QN）**：A2G-blind 7 维 + G2A-only MapGuide 教师 + 距离 TSP 顺序 + n-step/动作重复/DQfD 预训练（feat/mac 旧奖励），seed 1/2。结果见 §9。
- **Mac 已完成层面 3 的 TD3 系 baseline（A2G-blind，旧奖励）**：Yang-style+TD3 = 254.0s(102/120)、EEGA-TD3 = 279.5s(92/120)、Distance-TSP+TD3 = 311.0s(56/120)；A2G 可见消融 229.0s(91/120)，即去掉 A2G 恶化 +82s(+35.8%)。

## 7. 待办

- [x] 层面 1：D3QN 修复版结果确认（274.5s，53%）
- [ ] 层面 2：QoS 扫描（TD3，不同 T_out 阈值）
- [x] 层面 3：DRL-SA / SA-D3QN 已由 Win 用修复版 D3QN 完成（296.5s / 289.0s）；EEGA-TD3、Yang-style 已由 Mac 完成
- [ ] **奖励版本统一**：Win 已用 v5（0.5/MB），Mac baseline 用旧奖励；暂记录不处理，需后续统一
- [ ] 课程版 Proposed（IGA+TD3 low，Mac 在等，需 ≤229.0s）
- [ ] 配方口径定稿：`0.5→0`（course-final）vs `0.5→0.2`（legacy-0.2）
- [ ] 判断好坏一律看纯 actor `eval_best`（无引导、无噪声）

## 8. 原始论文实现规范（层面 3 baseline，来自 GPT 对四篇 PDF 的提取核验）

> 约定：每篇分「原文事实 / 未公开参数 / 本项目适配」。A2G-free = 不读取项目独立 A2G 地图（A2G-map-unaware），**不是**删除上传物理链路——环境上传速率与统一评价仍保留。

### 8.1 DRL-SA（2024）
- **原始目标**：最小化「通信中断时间 + 总巡检服务时间」的加权和。
- **序列**：SA，目标 = 总巡检服务时间（含数据采集/卸载，**不是**纯距离）。
- **轨迹**：**D3QN**（Dueling Double DQN，离散 8 个飞行方向）。
- **关键点**：原文轨迹模块是 D3QN，不是 TD3；用 Nm 步回报；每 Q 回合更新目标网络。
- **未公开参数**：SA 的 T0 / 终止温度 / 迭代数 Kt / 冷却率、D3QN 学习率 / γ / batch 等数值均「原文未给出」。

### 8.2 EEGA-TD3（2025）
- **原始目标**：最小化能耗。
- **序列**：EEGA（自适应遗传算法），加权距离矩阵 Γ，边权 `L_ij = Ψ·E_ij`，其中 `Ψ = I_oi / Q_ij`（数据量 / 上行吞吐量）。
  - 即 **边权 = 数据量 × 边能耗 / 吞吐量 ≈ 能耗 × 上传时长**。
- **轨迹**：TD3（连续动作，极坐标角度 + 速度，Δt = 0.5 s）。
- **未公开参数**：种群 / 交叉 / 变异概率 / 迭代数等「原文未给出」。

### 8.3 SA-D3QN（2024）
- **原始目标**：最小化通信中断时间（受电池能量约束）。
- **序列**：SA，目标 = 总飞行距离（恒速下能耗 ∝ 距离，见 P(Vi)×距离/Vi）。
- **轨迹**：D3QN（Dueling DQN，离散）。
- **关键点**：SA 目标是**纯距离 TSP**（不是服务时间）；D3QN 比 DDQN 好 15–25%。
- **未公开参数**：SA 温度/冷却率、D3QN 超参数「原文未给出」。

### 8.4 Yang（2024 IEEE TMC）
- **原始目标**：最小化总能耗（飞行 + 计算），UAV-MEC 框架，DVFS 计算能耗 `Ecop = κU·fU³(t)dt`。
- **遍历顺序**：基于「边估计吞吐量 vs 数据量」的加权因子（数据大时以数据卸载为主导）。
- **轨迹**：**SCA + BCD**（逐次凸近似 + 块坐标下降，迭代数学优化，**非学习**）。
- **关键点**：原文是 SCA/BCD，不是 TD3；`Yang-style+TD3` 是本项目适配（取 Yang 的加权遍历顺序 + TD3 轨迹）。

### 8.5 落地矩阵（5 个层面 3 baseline）

| baseline | 序列算法 | 轨迹算法 | A2G | 当前状态 |
|---|---|---|---|---|
| Distance-TSP + TD3 | 距离 TSP | TD3 | ❌ blind | Mac 训练中 |
| DRL-SA | SA（服务时间→A2G-free 下退化为距离） | **D3QN** | ❌ blind | **Win 完成 296.5s** |
| EEGA-TD3 | EEGA（数据×能耗/吞吐） | TD3 | ❌ blind | **Mac** |
| SA-D3QN | SA（距离 TSP） | **D3QN** | ❌ blind | **Win 完成 289.0s** |
| Yang-style + TD3 | 加权因子（吞吐 vs 数据） | TD3（适配） | ❌ blind | **Mac** |

> 分工：**Win 跑 DRL-SA、SA-D3QN**（都走 D3QN，我们本地已有 D3QN 训练管线）；**Mac 跑 EEGA-TD3、Yang-style+TD3**（都走 TD3）。

### 8.6 禁止误述

- DRL-SA 原文轨迹是 **D3QN**，不得写成 TD3。
- Yang 原文是 **SCA/BCD**，不得写成 TD3；`Yang-style+TD3` 必须声明为适配版本。
- EEGA/Yang 的吞吐量加权边权**不能简化成纯距离**。
- SA-D3QN 的 SA 目标是**最小距离**（恒速能耗），不得写成服务时间最小化。
- A2G-free ≠ 删除上传链路；只表示 baseline 不读取独立 A2G 地图。

## 9. D3QN 结论（已反转：修复后能学会）

### 9.1 关键转折：D3QN 修复版成功

Mac 在 feat/mac 实现的**三件套修复**（n-step 回报 n=10 + 动作重复 k=2 + DQfD 预训练 40 回合/2 万步）让 D3QN 从 0/120 变成**可用的离散基线**：

| 训练 | 配置 | 总成功率 | 末 1000 回合成功率 | eval_best T+τ |
|---|---|---|---|---|
| d3qn-v5-fixed | A2G-aware | 106/200（53%） | 67.5% | **274.5s**（ep1875） |
| sa-d3qn-fixed（SA-D3QN） | A2G-free | 93/200（46.5%） | 50.0% | **289.0s**（ep4850） |
| drl-sa-fixed（DRL-SA） | A2G-free | 53/200（26.5%） | 52.5% | **296.5s**（ep4800） |

### 9.2 正确的诊断：之前失败是"复现缺失机制"，不是"离散动作不行"

此前 0/120 的根因不是离散动作空间，而是我们漏掉了 **DRL-SA 原文的 n-step 回报**，加上没有**动作重复**与 **DQfD 预训练**：

- **n-step**：把完成/失败的终局奖励有效传回早期动作，解决长时域信用分配。
- **动作重复**：决策地平线减半，缩小"绕行"长序列的搜索空间。
- **DQfD 预训练**：在线交互前先用示范填满 Q 网络，给一个像样的起点。

（哪个最关键待消融；但三件套合起来让 D3QN 从"停在 1 点全超时"变成"稳定走完 6 点"。）

### 9.3 修正后的对比口径（层面 1，A2G-aware）

| 方法 | T+τ (s) | 说明 |
|---|---|---|
| A\*（oracle） | 214.0 | 非学习，最慢（高负载 463s 搜索） |
| **TD3（连续）** | **235.5** | 本文方法 |
| MapGuide | 273.0 | 弱规划参照 |
| D3QN（离散，修复版） | 274.5 | 补 n-step/动作重复/DQfD 预训练 |

结论：连续 TD3 仍优于离散 D3QN（274.5 vs 235.5），且 D3QN 只是接近 MapGuide 水平——**连续动作 + 行为克隆的 TD3 仍是首选**，但这不再是"离散学不会"，而是"离散学会但绕行粗、慢于连续"。

**层面 3 完整对比（全部 A2G-blind + 旧奖励）：**

| baseline | 轨迹 | eval_best T+τ (s) | 成功率 |
|---|---|---|---|
| Yang-style+TD3 | TD3 | 254.0 | 102/120 |
| EEGA-TD3 | TD3 | 279.5 | 92/120 |
| SA-D3QN（Win，修复版） | D3QN | 289.0 | 93/200 |
| DRL-SA（Win，修复版） | D3QN | 296.5 | 53/200 |
| Distance-TSP+TD3 | TD3 | 311.0 | 56/120 |

结论：D3QN 系基线（289–296.5）弱于最好的 TD3 系（Yang 254、EEGA 279.5），但好于最差的 TSP+TD3（311）；A2G-aware 的 D3QN（274.5）优于所有 A2G-free 的 D3QN，与"去掉 A2G 恶化"一致。

### 9.4 对层面 3 的影响

DRL-SA / SA-D3QN 的轨迹模块是 D3QN，其原文本就含 n-step 机制。用修复版 D3QN（含 n-step）复现后，**二者在 A2G-free 下均成功**（SA-D3QN 289.0s、DRL-SA 296.5s），不再是负结果。

## 10. Joint A* 内存上限问题（2026-11-07，Win → Mac 待议）

Mac §14 的正式 Joint A*（单状态全联合：节点+已访问集合+待传数据+连续中断+时间槽+当前目标）在三档都命中 2000 万 label 上限（约 1595/3057/3810 s），无完整路径。Win 侧看法与候选方案，请 Mac 定夺：

1. **降联合度**：不把"巡检顺序"放进 A* 状态，改用固定 IGA 顺序 + 逐段 A*（每段在给定起点下最优）。这等价于 Mac §13.3 的 legacy A*（216.0/316.5/445.5 s），但 §14.4 已禁止把它当正式 A*。若采用，需重新定名（如"顺序分段 A*"）并明确它不是联合最优。
2. **更强剪枝/分桶**：对"已访问集合"做集合-位掩码 + 支配哈希去重；数据/中断分桶更粗；启发式加剩余各段下界（此前 Win 在逐段 A* 上已用此法，节点数 4 万～65 万，成功）。
3. **状态空间降维**：去掉"当前目标选择"维度，改为固定顺序后只剩轨迹搜索（即方案 1）。
4. **时间/内存换算**：20M label 上限是 3 档全挂的直接原因，若提到 2 亿 label（内存 ~几十 GB），可能可行但代价高。

Win 倾向方案 1（改回固定 IGA 顺序 + 逐段 A*，作为"非学习图搜索基线"），配合方案 2 的启发式保证逐段最优。请 Mac 确认最终 A* 定位与措辞口径，否则 Abstract/结论与 baseline 排名无法定稿。

## 11. PPO 配方修正（2026-11-07，Win 执行方案 B）

Mac 在 `WIN_OPTIONAL_DRL_MATRIX.md` 中指定的 PPO 配方只有「一次性 BC 预训练（20 回合 / 5000 步）」，在线训练全程无示范。Win 观察到 PPO 在 low 档 0/120、medium/high 档同样 0 成功，与当初 D3QN 修复前同根因：示范供给不足。

为保证 PPO 与 TD3/D3QN 的公平对比，Win 已按「方案 B」把 PPO 改为同等持续示范机制：

| 机制 | 原配方（Mac 指定） | 修正后（与 TD3/D3QN 对齐） |
|---|---|---|
| 预训练 | 20 回合 / 5000 步 | **40 回合 / 20000 步** |
| 在线 MapGuide 代飞 | 无 | `guide_prob 0.5 → 0.2`（前 400 回合衰减后保持） |
| 持续示范回放 | 无 | 持久 demo buffer，每批混 25% |
| BC 辅助损失 | 无 | `bc_weight 1.0 → 0.25`（heading+cos + distance，与 TD3 同式） |

- 代码改动：`PPO.py`（update 增加 demo/bc_weight/demo_fraction 的 BC 辅助损失）、`Train_PPO.py`（新增 guide_prob/guide_min_prob/guided_warmup_episodes/demo_fraction/bc_weight/bc_min_weight/bc_decay_episodes 参数，重构在线循环支持整回合 MapGuide 代飞 + 持久 demo buffer；config 记录 guide_schedule/bc_schedule/demo_sampling_fraction）。
- 旧配方 PPO low 的 0/120 结果作废，不作为「PPO 差于 TD3」的正式证据；medium/high 正在用新配方重跑。
- 请 Mac 知悉此偏离：若坚持原配方，PPO 的失败只能解读为「示范供给不足」，不能解读为「连续动作 DRL 中 TD3 优于 PPO」。

## 12. Win 实验进度（2026-11-07）

- ✅ 受控顺序消融（low，A2G-aware，3 组）已完成：Distance-TSP 235.0 < EEGA 243.5 < IGA 244.5 < Yang 247.5 → IGA 无独立增益，倾向「分支 B」。
- ✅ 14 组矩阵中 Layer-1 D3QN medium/high 已完成。
- 🔄 剩余 12 组训练中（3 路并发）：PPO medium/high（新配方）、TD3 基线 6 组（distance-tsp/eega/yang × med/high，A2G-blind）、D3QN 基线 4 组（drl-sa/sa-d3qn × med/high，A2G-blind，5000 回合）。
- 冻结顺序已核验与 `WIN_OPTIONAL_DRL_MATRIX.md` §2 逐位一致（`freeze_baseline_orders.py` 已跑，`v4-sequence-orders/orders.json` 已生成）。

## 13. 全量实验结果汇总（2026-11-08，Win → Mac）

### 13.1 受控顺序消融（low，A2G-aware，3 组新跑 + IGA 复用）

| 顺序 | eval_best T+τ | 结论 |
|---|---|---|
| Distance-TSP | 235.0 | 最优 |
| EEGA | 243.5 | |
| IGA（复用） | 244.5 | |
| Yang | 247.5 | 最差 |

→ **IGA 无独立增益**（Distance-TSP 反而更好），支持 PAPER_STRATEGY_V4 §10 的「分支 B」。
（注：Mac 在论文仓 sequence-ablation.json 记录的数字与此略有差异：236.0/247.0/244.0，EEGA 与 Yang 排序翻了个，疑为 Mac 用自己评估代码重新回放 eval_best 所致，建议 Mac 核对来源。）

### 13.2 14 组矩阵 + PPO low 结果

| 方法 | 档位 | 状态 | 成功率 | eval_best T+τ |
|---|---|---|---|---|
| TD3（A2G-aware，复用） | low/med/high | ✅ | 108/92/75 | 244.5 / 352.0 / 498.5 |
| D3QN layer1（A2G-aware） | medium | ✅ | 20/120 | 400.0 |
| D3QN layer1（A2G-aware） | high | ✅ | 13/120 | 532.5 |
| PPO（新配方，A2G-aware） | low | ✅ | 4/120 | 288.5 |
| PPO（新配方） | medium | ✅ | 2/120 | 562.0 |
| PPO（新配方） | high | ❌ | 0/120 | 不可行 |
| TD3 distance-tsp（A2G-blind） | medium | ✅ | 21/120 | 658.5 |
| TD3 eega（A2G-blind） | medium | ✅ | 21/120 | 658.5 |
| TD3 yang（A2G-blind） | medium | ✅ | 25/120 | 559.0 |
| TD3 三基线（A2G-blind） | high | ❌ 预检失败 | — | 不可行 |
| D3QN drl-sa（A2G-blind） | medium | ✅ | 4/200 | 649.5 |
| D3QN sa-d3qn（A2G-blind） | medium | ✅ | 2/200 | 719.5 |
| D3QN 两基线（A2G-blind） | high | ❌ | 0 | 不可行 |

### 13.3 三个核心结论

1. **A2G-aware TD3 全面最优**，medium 档与 A2G-blind 基线拉开 37–104% 差距（352 vs 559–719s）。
2. **高负载 A2G-blind 彻底不可行**：TD3 三基线预检失败（G2A-only 教师 1500 MB 下 500s 超时），D3QN 两基线训练但 0 成功 → A2G 地图在高负载下更关键。
3. **PPO 给了与 TD3/D3QN 同等的持续示范后仍学不好**（low 4/120、medium 2/120、high 0）→「连续 DRL 中 TD3 优于 PPO」成立且非示范差异所致。

### 13.4 回传文件

- `journal_results/v4-sequence-ablation/`（顺序消融）
- `journal_results/v4-models/layer1/d3qn/{medium,high}`、`v4-models/ppo/A/{low,medium,high}`
- `journal_results/v4-baselines/{distance-tsp,eega,yang,drl-sa,sa-d3qn}/...`
- 全部含 run_config / eval_history / training_history（或 UAV_TD3_improved）/ preflight / 训练图；已排除 models/ 与 logs/。
- 注：TD3 high 三基线无 eval_history（预检即失败），其 preflight.json 记录了失败现场。

## 14. 高负载 A2G-blind 基线 horizon 扩展 + 口径同步（2026-11-08）

### 14.1 高负载 A2G-blind 失败根因与处理

高负载（1500 MB）A2G-blind 基线失败根因是**超时而非中断**：G2A-only 教师/策略在 500s（1000 时隙）内传不完 1500 MB（预检只到 5/6 点、1209/1500 MB，最长连续中断仅 2.5s < 6s 预算）。

按用户指示，**把高负载 A2G-blind 基线 horizon 扩到 2000 时隙（1000s）重跑**，保证能完成；完成后其 T+τ 应劣于我方 A2G-aware TD3（498.5s）。冒烟已验证：2000 时隙下预检成功（672.5s 完成 1500 MB）。

重跑清单（5 组，3 路并发，进行中）：
- TD3 distance-tsp / eega / yang high（A2G-blind，3000 episodes）
- D3QN drl-sa / sa-d3qn high（A2G-blind，5000 episodes）

代码：`Train_SINR.py` / `Train_D3QN.py` / `evaluate_policy.py` 新增 `--total_steps`（默认 1000）。

### 14.2 PPO 实际代码已补推

PPO 实际代码（含持续示范「方案B」）已提交并推送：`feat/mac @ fab24af`，含：
- `PPO.py` / `Train_PPO.py`：guide_prob/demo_fraction/bc_weight 持续示范机制，预训练 40 回合/20000 步；
- `Train_SINR.py` / `Train_D3QN.py` / `evaluate_policy.py`：`--total_steps` horizon 参数化。

### 14.3 口径对齐（按 Mac 要求）

1. **顺序消融暂时保留 Mac 口径**：论文仓 sequence-ablation.json 的数字（236.0/247.0/244.0）暂用，不用 Win 侧 eval_best（235.0/243.5/247.5）。
2. **PPO 新配方只有在代码、测试和复现实验补齐后，才正式替换旧结果**（当前 PPO 三档新配方结果仅作内部参考，不覆盖旧 0/120 口径）。
