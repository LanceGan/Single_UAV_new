# Baseline 对比方案与实施计划

> 更新时间：2026-10-31  
> 项目类型：课程作业  
> 适用场景：固定 Low 异构数据量、双无线电地图 UAV 巡检

---

## 1. 文档目的

本文档确定课程作业最终采用的Baseline、统一实验条件、方法命名、当前实现状态和实施优先级。

当前目标不是完整复现所有参考论文，而是在同一UAV巡检环境中构造具有代表性的顺序优化、连续DRL、离散DRL和图搜索对比。

所有适配方法必须明确哪些部分来自参考文献，哪些部分为了统一当前场景进行了简化。

---

## 2. 优化问题与统一场景

所有方法统一优化：

\[
\min T+\tau_{\mathrm{total}},
\]

其中：

- \(T\)：完整巡检任务完成时间；
- \(\tau_{\mathrm{total}}\)：所有存在待传数据的时隙累计时长。

QoS为硬约束：

\[
T_{\mathrm{out}}\leq100\ \mathrm{s}.
\]

统一场景：

- 相同2 km × 2 km区域；
- 相同G2A/A2G固定无线电地图；
- 相同4个GBS、6个巡检点、起点和终点；
- 相同50 m/s最大速度、0.5 s时隙和25 m单步最大距离；
- 相同500 s任务上限；
- 相同上传速率计算、到点半径和失败条件；
- 相同Low异构数据向量：`[90, 90, 60, 60, 90, 60] MB`。

---

## 3. 最终方法矩阵

### 3.1 主方法

| 论文名称 | 顺序优化 | 轨迹优化 | 当前状态 |
|---|---|---|---|
| Proposed | Data-aware IGA | TD3 | 已实现 |

主方法中的TD3在训练前期使用MapGuide可行示范，教师概率在400轮内降为0；测试阶段为纯Actor。

### 3.2 核心Baseline

| 编号 | 论文显示名称 | 顺序优化 | 轨迹优化 | 优先级 |
|---|---|---|---|---|
| B1 | EEGA-TD3 (GA proxy) | 普通GA代理 | TD3 | 高 |
| B2 | DRL-SA | SA | 连续DRL/TD3 | 高 |
| B3 | Distance-TSP+TD3 | 精确距离TSP | TD3 | 最高 |
| B4 | D3QN | Data-aware IGA | D3QN | 已实现 |
| B5 | A* | Data-aware IGA | QoS约束A* | 已实现 |
| B6 | DDPG | Data-aware IGA | DDPG | 已实现 |

### 3.3 暂缓方法

| 方法 | 处理 |
|---|---|
| SA-D3QN | 若时间充足再将D3QN与SA顺序组合；当前先使用Data-aware IGA顺序 |
| Yang-style+TD3 | 暂缓；未取得原始边权公式前不得实现 |
| 完整EEGA-TD3 | 暂缓；需获得原文公式后替换GA代理 |

---

## 4. EEGA-TD3 的临时代理方案

### 4.1 命名规则

当前不能直接把普通GA宣称为已完整复现EEGA。

代码、实验目录和结果表建议使用：

```text
EEGA-TD3 (GA proxy)
```

中文写法：

```text
EEGA-TD3（普通GA代理）
```

若课程报告图例空间有限，可以写 `GA-TD3`，并在正文说明它暂时代替EEGA-TD3。

禁止写：

> “本文完整复现了参考文献中的EEGA。”

### 4.2 临时实现

- 使用普通遗传算法的选择、交叉和变异机制；
- 固定起点与终点，染色体只排列6个巡检点；
- 使用统一场景与Low数据向量；
- 轨迹阶段复用与Proposed完全相同的TD3架构、MapGuide早期示范、训练预算和评估流程；
- 仅改变顺序优化模块，避免把轨迹网络差异混入比较。

普通GA的fitness必须在代码和报告中明确记录。若当前只使用几何距离，则该方法本质上接近启发式Distance-TSP，不能声称包含EEGA的能耗—吞吐量机制。

### 4.3 后续替换条件

获得EEGA论文正文后，需要核对：

- 边权/适应度公式；
- 能耗模型；
- 吞吐量和数据任务的权重；
- 自适应交叉率与变异率；
- 精英保留策略；
- 终止条件。

替换完成前，所有结果必须保留“GA proxy”标记。

---

## 5. DRL-SA

### 5.1 目标

使用Simulated Annealing决定巡检顺序，再使用统一连续DRL优化轨迹。

### 5.2 当前代码状态

- 会议论文和历史绘图中存在DRL-SA结果；
- `DJI_origin.py`存在供SA调用的旧接口；
- 当前仓库没有独立、可复现并与新World对齐的SA训练入口；
- `PSO.py`是粒子群算法，不能重命名为SA。

### 5.3 实施要求

- 新建独立SA顺序求解器；
- 固定起点、终点和6个巡检点；
- 使用与参考方法一致的邻域交换、温度衰减和Metropolis接受机制；
- SA只负责顺序，不修改World上传和QoS规则；
- 轨迹阶段复用统一TD3设置；
- 输出实际访问顺序、SA目标值和随机种子。

---

## 6. Distance-TSP+TD3

这是最优先补充的Baseline。

### 6.1 顺序目标

\[
\min_O\sum_{(i,j)\in O}d_{ij}.
\]

### 6.2 实现

- 不使用A2G地图；
- 不使用G2A地图；
- 不使用数据量；
- 枚举6个巡检点的720种排列；
- 加入固定起点与终点；
- 精确选择总几何距离最短的顺序；
- 将该顺序送入与Proposed相同的TD3训练入口。

### 6.3 对比意义

用于验证异构数据量和双无线电地图是否真的改善巡检顺序，而不是仅靠TD3轨迹优化获得性能提升。

---

## 7. D3QN

当前D3QN使用：

- 8维变体A状态；
- 33个唯一离散动作；
- Dueling网络；
- Double-DQN目标；
- epsilon-greedy；
- 早期离散MapGuide示范；
- 教师概率在400轮后归零；
- 独立示范池和25%示范batch；
- 学习率 `1e-4`。

论文图例写 `D3QN`。方法章节说明其前期使用离散MapGuide可行示范，测试阶段不使用教师。

---

## 8. DDPG

DDPG使用与TD3相同：

- 状态；
- 连续动作；
- MapGuide示范；
- ReplayBuffer；
- 训练回合数；
- 纯Actor评估规则。

仅网络更新机制不同。若DDPG因单Critic高估导致无可行结果，应如实报告失败，不得给失败轨迹计算可行排名。

---

## 9. A*图搜索

A*已经实现并完成Low负载World重放：

| 指标 | 数值 |
|---|---:|
| \(T\) | 135.0 s |
| \(\tau\) | 79.0 s |
| \(T+\tau\) | 214.0 s |
| \(T_{out}\) | 27.5 s |
| 上传量 | 450 MB |

A*采用顺序分段搜索，`optimality_proven=false`。报告中称为QoS约束A*基线或顺序分段A*，不得称为整个任务的全局最优解。

A*没有训练奖励曲线，只参加最终指标表。

---

## 10. 统一训练配置

连续DRL统一：

| 参数 | 数值 |
|---|---:|
| 数据量 | Low |
| episodes | 3000 |
| seed | 1 |
| gamma | 0.995 |
| network width | 256 |
| replay capacity | 200000 |
| batch size | 256 |
| QoS limit | 100 s |
| eval interval | 25 episodes |
| MapGuide初始概率 | 0.5 |
| MapGuide最终概率 | 0 |
| 教师衰减时长 | 400 episodes |
| BC权重下限 | 0.25 |
| 示范batch比例 | 0.25 |

D3QN使用独立配置，但总episodes、seed、环境、任务和评估间隔保持一致。

---

## 11. 公平性要求

所有方法必须使用：

- 相同地图哈希；
- 相同Low数据向量；
- 相同起终点与巡检点；
- 相同运动约束；
- 相同上传速率计算；
- 相同100 s QoS硬约束；
- 相同500 s回合上限；
- 相同最后一份数据上传规则；
- 无探索纯策略最终评估。

优化指标只在可行解中比较。提前终止、QoS失败和超时轨迹不能因为时间较短而获得更好排名。

---

## 12. 统一评价指标

最终结果表至少包含：

| 指标 | 含义 |
|---|---|
| Success | 是否完成全部巡检、上传并到达终点 |
| \(T\) | 完成时间 |
| \(\tau_{total}\) | 累计待传时延 |
| \(T+\tau_{total}\) | 优化目标 |
| \(T_{out}\) | G2A累计中断时间 |
| Flight distance | 实际飞行距离 |
| Runtime | 算法训练或规划运行时间 |

DRL额外输出奖励曲线；A*只输出最终指标。

---

## 13. 奖励曲线展示规则

TD3在2000轮后存在策略退化。主文可以展示0–2000轮，但必须遵守：

1. 图题或正文注明“展示前2000轮训练过程”；
2. 不声称2000轮后仍稳定；
3. 保留完整3000轮原始数据；
4. 最终指标仍由完整训练过程中无噪声 `eval_best` 选择；
5. TD3/DDPG只统计 `ep_guided=False` 的Actor回合；
6. D3QN使用非教师回合奖励；
7. 若三条算法放在同图中，统一使用0–2000横轴范围；
8. 推荐200个Actor回合移动平均，不删除失败区间。

可以在正文写：

> 为清晰展示主要学习阶段，奖励曲线给出前2000个训练回合；最终策略根据整个训练过程中周期性无探索评估得到的最佳可行模型确定。

---

## 14. 实施优先级

### 第一优先级

1. 完成正在运行的TD3、DDPG、D3QN Low实验；
2. 生成三种DRL的0–2000轮最终奖励曲线；
3. 填写TD3/DDPG/D3QN/A*最终指标表。

### 第二优先级

4. 实现Distance-TSP顺序；
5. 使用统一TD3训练Distance-TSP+TD3；
6. 实现可复现的SA顺序；
7. 使用统一TD3训练DRL-SA。

### 第三优先级

8. 使用普通GA构建 `EEGA-TD3 (GA proxy)`；
9. 在获得EEGA原文公式后决定是否替换；
10. 若时间仍充足，再考虑SA-D3QN。

Yang-style方法本课程阶段不实现。

---

## 15. 当前实现状态审计

| 模块 | 状态 |
|---|---|
| Proposed Data-aware IGA+TD3 | 已实现 |
| DDPG | 已实现，训练中 |
| D3QN | 已实现，加入离散MapGuide后待重训 |
| A* | 已实现，Low结果完成 |
| Distance-TSP顺序 | 尚缺当前统一入口 |
| DRL-SA | 旧代码/旧结果存在，当前统一入口缺失 |
| EEGA-TD3 | 未实现；暂用普通GA代理方案 |
| SA-D3QN | 未实现SA组合 |
| Yang-style | 未实现且暂缓 |

---

## 16. 结果表模板

| Method | Sequence | Success | \(T\) | \(\tau\) | \(T+\tau\) | \(T_{out}\) |
|---|---|---:|---:|---:|---:|---:|
| Proposed | Data-aware IGA | 待填 | 待填 | 待填 | 待填 | 待填 |
| EEGA-TD3 (GA proxy) | Ordinary GA | 待填 | 待填 | 待填 | 待填 | 待填 |
| DRL-SA | SA | 待填 | 待填 | 待填 | 待填 | 待填 |
| Distance-TSP+TD3 | Distance TSP | 待填 | 待填 | 待填 | 待填 | 待填 |
| DDPG | Data-aware IGA | 待填 | 待填 | 待填 | 待填 | 待填 |
| D3QN | Data-aware IGA | 待填 | 待填 | 待填 | 待填 | 待填 |
| A* | Data-aware IGA | Yes | 135.0 | 79.0 | 214.0 | 27.5 |

---

## 17. 写作时禁止的表述

- 不得把普通GA写成已完整复现EEGA；
- 不得把PSO称为SA；
- 不得把顺序A*称为全局最优；
- 不得把失败轨迹的提前终止时间当作可行指标；
- 不得把MapGuide教师回合当作TD3/DDPG自主性能；
- 不得隐藏完整训练数据并声称后期稳定；
- 不得在没有原论文公式时编造EEGA或Yang-style边权。

---

## 18. 参考文献映射

1. EEGA-TD3：Wang et al., *Trajectory design of cellular-connected UAV patrol and mobile edge computing system: A deep reinforcement learning approach*, Computer Networks, 2025, DOI: `10.1016/j.comnet.2025.111384`。
2. DRL-SA：Wang et al., *Trajectory Optimization for Connectivity-Aware Inspection UAV: A Hybrid Algorithm of DRL and SA*, PIMRC 2024, DOI: `10.1109/PIMRC59610.2024.10817221`。
3. SA-D3QN：Wang et al., *Optimization of Communication Quality for Energy-Limited Inspection AAV: A Hybrid Algorithm*, IEEE OJ-COMS, 2024, DOI: `10.1109/OJCOMS.2024.3507818`。
4. Yang-style：Yang et al., *Energy Efficient Transmission Strategy for Mobile Edge Computing Network in UAV-Based Patrol Inspection System*, IEEE TMC, 2024, DOI: `10.1109/TMC.2023.3315477`。

