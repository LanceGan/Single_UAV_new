# V5 Experiment Assignment

更新日期：2026-11-08

## 1. 立即执行

若 Win 上仍有旧版 v4 实验在运行，全部停止：

- `Train_SINR.py`
- `Train_D3QN.py`
- `Train_PPO.py`
- v4 baseline / QoS / weight / sequence 队列脚本
- 任何仍使用 `N=1000` slots（500 s）或 QoS 违约立即终止的训练

不要删除已有结果、日志或 checkpoint：

- 已有结果统一视为 `legacy-v4`，只用于追溯和失败分析；
- 不允许把 v4 与 v5 数据混入同一主表；
- 唯一例外：Low 档 Distance-TSP 的 A2G ablation 暂按用户决定复用，
  但在最终数据清单中必须标明其训练协议来源。

Win 在收到 v5 代码的明确 commit 和启动指令前，不启动新训练。

## 2. V5 统一协议

| 项目 | 设置 |
|---|---|
| 任务时域 | `N=2000` slots = 1000 s |
| QoS | `T_out^cont <= 6 s` |
| 中断概率阈值 | `kappa_out=0.5` |
| QoS 违约处理 | 记录违约，但继续任务 |
| 完成定义 | 6/6 inspection points + full upload + reach end |
| 约束定义 | `Complete=true` 且 `T_out^cont <= 6 s` |
| 学习策略评估 | actor-only，无 MapGuide，无探索噪声 |
| 主结果 | `T`, `tau_total`, `T+tau_total`, `T_out^cont` |
| 失败结果 | 完整但违约也保留目标值，Constraint 标 `x` |
| seed | seed 1；不报告 seed-level interval |

模型选择顺序：

1. 完整且约束可行：最小 `T+tau_total`；
2. 无可行策略但存在完整任务：完整任务中最小 `T+tau_total`；
3. 没有完整任务：最大巡检点和上传量的诊断轨迹。

## 3. P0 代码与验收门槛

以下项目全部通过后才能启动正式矩阵：

1. `total_steps` 可从 CLI、config 和 evaluator 一致传递；
2. QoS 违约不再提前结束任务；
3. `mission_complete` 与 `qos_feasible` 独立记录；
4. 支持“完整但违约”的模型选择；
5. 新增 SAC；
6. DDPG、PPO、SAC 使用与 TD3 对应的统一示范/BC 接口；
7. 补齐 Win 上 PPO 新配方的实际源码和测试；
8. 新增 sequence-only PSO 与 A*；
9. 新增顺序代理指标：平均 A2G rate、平均/最大 G2A outage；
10. 重新设计 PRIGA 通信感知 edge cost；
11. run config、history、actor、trajectory 和 summary 格式一致；
12. 单元测试和 100 回合 smoke test 通过。

## 4. 实验矩阵

| ID | 实验 | 固定条件 | 对比 | 网格 | 结果 |
|---|---|---|---|---|---|
| E1 | Trajectory executor | PRIGA, A2G-aware, 6 s | CA-TD3, D3QN, DDPG, PPO, SAC | 5 x 3 | Low figures; M/H table |
| E2 | Inspection order | Same CA-TD3, A2G-aware, 6 s | PRIGA, GA, PSO, SA, A*, EEGA-style, Yang-style | 7 x 3 | Low proxy table; all-tier E2E table |
| E3 | A2G ablation | Distance-TSP, 6 s | A2G-blind / A2G-visible | 2 x 3 | Low reuse; M/H 4 new runs |
| E4A | QoS scan | PRIGA+CA-TD3, balanced weight | 3/4/5/6/8 s | 3 x 5 | tables + representative trajectories |
| E4B | Weight scan | PRIGA+CA-TD3, 6 s | (0.8,0.2), (0.5,0.5), (0.2,0.8) | 3 x 3 | tables + 3 trajectories/tier |
| E5 | Literature baselines | V5 common protocol | DRM-HO, MCO-A*, EEGA-TD3, Distance-TSP+TD3, Yang-style+TD3, DRL-SA, SA-D3QN | 7 x 3 | Low trajectories+table; M/H table |

Distance-TSP 不进入 E2 正式顺序比较，只在 E3 作为固定顺序的消融控制，
并在 E5 作为适配的端到端文献基线。

## 5. 输出规范

### E1

Low：

- raw reward + moving average；
- deterministic feasibility curve；
- 五个方法在 A2G map 上的 eval-best trajectory；
- 五个方法在 G2A map 上的同一 trajectory。

Medium/High：

| Method | T | tau | T+tau | T_out_cont | Complete | Constraint |
|---|---:|---:|---:|---:|---|---|

### E2

Low sequence proxy：

| Method | Order | Mean A2G rate | Mean leg outage | Max outage | Estimated T | Estimated tau |
|---|---|---:|---:|---:|---:|---:|

All tiers end-to-end：

| Method | T | tau | T+tau | T_out_cont | Complete | Constraint |
|---|---:|---:|---:|---:|---|---|

E2 不生成 trajectory figure。

### E3

| Level | Visibility | T | tau | T+tau | T_out_cont | Complete | Constraint |
|---|---|---:|---:|---:|---:|---|---|

### E4A / E4B

完整数值表必须包含：

- level；
- QoS budget 或 objective weights；
- `T`；
- `tau_total`；
- `T+tau_total`；
- `T_out^cont`；
- feasible rate；
- Complete；
- Constraint。

轨迹图：

- E4A 每个 level 仅画 3/6/8 s 三条代表轨迹；
- E4B 每个 level 画三个权重对应轨迹。

### E5

Low 包含七方法 eval-best 轨迹与完整指标；Medium/High 只出表。

| Method | Reference | T | tau | T+tau | T_out_cont | Visited/Upload | Complete | Constraint |
|---|---|---:|---:|---:|---:|---|---|---|

## 6. 分工

### Win（CUDA）

- E1 三档全部 DRL 训练；
- E2 七种顺序对应的三档 CA-TD3 训练；
- E3 Medium/High 四组；
- E4A/E4B v5 训练；
- E5 Medium/High 学习型基线；
- 每次只启动经过 Mac 审核的 config；
- 训练后回传 config、history、eval history、actor 和 eval-best trajectory。

### Mac

- 全部 P0 代码和测试；
- PRIGA 重新设计；
- sequence-only PSO/A* 和代理指标；
- v5 配置生成器与输出 schema；
- A2G Low legacy 数据审计；
- trajectory 和论文图表；
- 数据来源、失败分类和表格一致性审计；
- 给 Win 提供固定 commit、命令和验收脚本。

## 7. 执行波次

### Wave 0：止损与代码冻结

- Win 停止所有 v4 训练；
- 保留并标记 legacy-v4；
- Mac 完成 P0 代码；
- 冻结 v5 协议、schema、seed、horizon。

### Wave 1：Low pilot

先跑：

- E1 Low 五方法；
- E2 Low 七种顺序；
- 每组至少一个完整 6/6 + full-upload actor-only rollout。

通过条件：

- 输出 schema 完整；
- reward/feasibility curve 可由原始数据重画；
- 完整但 QoS 违约的方法保留 `T`, `tau`, `T+tau` 和 Constraint=x；
- CA-TD3 与 baseline 使用同一个 horizon 和评价器。

如果某个方法无法完成，只允许调整对所有方法一致的示范课程后重跑 pilot，
不允许单独特调某一个算法。

### Wave 2：E1/E2 全量

- E1 Medium/High；
- E2 Medium/High；
- 同步生成 Low 图表和全量表格；
- 完成 PRIGA 代理指标与端到端结果一致性审计。

### Wave 3：E3/E4

- E3 Medium/High 四组；
- E4A/E4B 稀疏扫描；
- 生成代表性 trajectory 图。

### Wave 4：E5 外部基线

- 先跑 Medium；
- 再跑 High；
- 在 1000 s 下允许 QoS 违约后继续；
- 所有方法使用统一 `Complete` 和 `Constraint` 字段。

### Wave 5：定稿与 QA

- 统一 JSON/CSV 汇总；
- 检查所有运行 provenance；
- 禁止 v4/v5 混表；
- 重新生成论文表格和图；
- 编译、字体、碰撞和源数据 QA。

## 8. 数据复用规则

可以直接引用但必须标注：

- Low Distance-TSP A2G-blind / visible 旧结果；
- 确定性地图 hash 和系统参数。

不得直接混用：

- v4 的 500 s 学习型训练结果与 v5 的 1000 s 训练结果；
- Mac 与 Win 的 sequence-ablation 数值；
- 旧 PPO 0/120 与新 PPO 结果；
- 旧累计中断和 v4/v5 最长连续中断。

## 9. 风险边界

- 不能预设“CA-TD3、PRIGA 必然最优”；结果必须由 v5 数据决定。
- 删除 Distance-TSP 只表示它不进入正式 E2 比较，不代表它在历史实验中
  没有取得更好的 Low 结果。
- EEGA-style 与 Yang-style 是独立顺序算法；如果作为比较项，必须使用其
  公开或明确声明的边权定义。
- 所有 adaptive baseline 必须在论文中标记 `adapted`。
