# 实验缓存优化待办

> 记录日期：2026-11-05
> 状态：等待当前 DRL-SA / SA-D3QN 队列结束后实施
> 原则：缓存只能减少重复计算，不得改变策略、轨迹、QoS 判定或随机序列

## 1. 实施时机

- 不在当前正在运行的 DRL-SA/SA-D3QN 进程中热修改代码。
- 不抢占当前 `run_mac_d3qn_baselines.sh` 和自动等待队列。
- 等当前队列结束后，单独建立缓存优化提交，再决定是否用于下一批正式实验。
- 缓存版本若被用于正式结果，`run_config.json` 必须记录缓存版本或提交号。

## 2. 优先级

### P1：MapGuide 静态图和路径缓存

- 缓存八邻接图的 CSR 和反向 CSR；
- 缓存 G2A 中断布尔图与 A2G 速率图；
- 缓存可由 `(source, target, blocked, transmit)` 唯一确定的最短路结果；
- 缓存键必须包含地图哈希、目标位置、动作边界和 `use_a2g`；
- 不能忽略噪声导致的 off-route 重规划。

验收：

- 同一世界状态下，缓存前后 route 和 action 逐位一致；
- 至少比较 transmit / upload 两种模式；
- MapGuide 训练 smoke 完成且 eval_history 指标一致。

### P1：MapGuide 基准示范轨迹

- 对固定 `(level, order, a2g_blind, map_hash)` 预生成确定性示范轨迹；
- 保存状态、动作、时间、QoS 和运行元数据；
- 使用前校验地图哈希、顺序和代码版本；
- 带噪声示范仍保留一致的局部重规划语义。

验收：

- 缓存回放与实时 MapGuide 的到达点、累计奖励和终止状态一致；
- 不允许用旧累计中断版本的示范缓存。

### P2：IGA 顺序和预检缓存

- 保存 low/medium/high 的 IGA 顺序、序列目标值和 exhaustiveness 校验；
- 保存 MapGuide 预检摘要与地图哈希；
- 训练启动时只校验缓存，不重复运行 500 代 GA 和完整预检。

### P2：地图与地形数组进程级缓存

- `FixedRadioMaps` 按文件路径和 mtime/哈希缓存数组；
- 缓存 `Rural_world.HeightMapMatrix`；
- 缓存地图 SHA256，避免每个 World 重算。

### P3：训练日志降噪

- `World.step_inside` 增加 `verbose` 控制；
- 默认只输出每 50 回合的摘要、周期评估和异常；
- 控制台信息减少不得改变 RNG 调用顺序或训练数据。

## 3. 禁止缓存

- 旧累计中断口径的模型、轨迹和 replay buffer；
- 不同 QoS 预算之间共享训练模型；
- 不同序列或不同 `a2g_blind` 设置之间共享状态-动作缓存；
- 未经地图哈希和代码版本校验的 MapGuide 路径。

## 4. 完成定义

- 缓存命中/未命中前后同一 seed 产生逐位一致的 actions 和 metrics；
- 完成全量测试；
- 提供基准速度测试，记录 cache on/off 的每回合耗时；
- 正式结果配置中可追溯到缓存版本。
