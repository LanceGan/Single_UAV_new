# 最长连续中断版本：全量重跑分工

> 更新日期：2026-11-05
> 前置版本：`feat/mac` 中 `state_schema = v4-max-contiguous`、`qos_constraint = max_contiguous_outage`
> 旧结果状态：所有累计中断口径结果标记为 `legacy-cumulative`，不得进入新版主表或主图

## 1. 结论

系统状态、QoS 可行域、episode 终止条件和 A* 剪枝均已改变。旧 TD3/D3QN 检查点不能复用，所有神经网络对比方法和所有依赖 QoS 可行性的确定性方法都要在新版代码上重跑。

新版输出统一使用独立根目录：

```text
journal_results/v4-qos/
journal_results/v4-models/
journal_results/v4-baselines/
journal_results/v4-deterministic/
journal_results/v4-summary/
```

不得覆盖或复用：

- `journal_results/qos-sweep/`
- `journal_results/qos-sweep-reweighted/`
- Win 在 2026-11-05 用累计中断口径启动的 QoS 结果

## 2. Win：GPU 神经网络训练

Win 负责 QoS、PPO、Proposed 中高档、层面 1 D3QN、三个 TD3 基线和 A2G
消融；DRL-SA 与 SA-D3QN 两个 D3QN 实验族由 Mac 执行。每组都必须满足：

```text
state_schema starts with "v4-max-contiguous"
qos_constraint = "max_contiguous_outage"
```

### 2.1 固定配方

- 数据档位按实验指定；
- Variant A；
- TD3 或 D3QN；
- seed 1；
- TD3、PPO 和层面 1 D3QN 使用 3000 episodes，层面 3 D3QN 使用 5000 episodes；
- 每 25 episodes 做一次纯 Actor 确定性评估；
- 最终只选择可行 `eval_best`；
- 固定地图、起终点、运动约束、上传规则。

### 2.2 Qos 网格：15 组 TD3

| 权重 | 预算 |
|---|---|
| `(0.8,0.2)` | 3, 4, 5, 6, 8 s |
| `(0.5,0.5)` | 3, 4, 5, 6, 8 s |
| `(0.2,0.8)` | 3, 4, 5, 6, 8 s |

输出根目录：

```text
journal_results/v4-qos/<weight>/qos<budget>
```

完整命令和验收规则见 `QOS_RETRAINING_PLAN.md`。

以下结果不要重复训练，直接复用：

- `w50-50/qos6` 同时作为 Proposed 低负载主结果；
- `w80-20/qos6`、`w50-50/qos6`、`w20-80/qos6` 同时作为三组权重实验。

8 s 结果只作为 QoS 轴的非束缚端点，不用于主方法和 baseline 比较。

### 2.3 数据量敏感性 TD3

在非束缚预算 6 s 下新增：

- Proposed TD3，`medium`；
- Proposed TD3，`high`；
- 低负载复用 `v4-qos/w50-50/qos6`。

输出：

```text
journal_results/v4-models/load/medium/
journal_results/v4-models/load/high/
```

### 2.4 层面 1 的 D3QN

在低负载、6 s 下重训 A2G-aware D3QN：

- episodes：3000（与 Proposed TD3 同步）；
- teacher：MapGuide；
- `guide_prob=0.5`，`guide_min_prob=0.5`；
- `n_step=10`；
- `action_repeat=2`；
- DQfD 预训练 40 回合、20000 updates；
- learning rate `1e-4`；
- epsilon decay 300000 steps；
- seed 1，eval interval 25。

`guide_min_prob=0.5` 表示 MapGuide 概率在预训练/在线训练阶段保持 0.5，
不再衰减到 0.2。评估仍然是纯 argmax、无 MapGuide。

输出：

```text
journal_results/v4-models/layer1/d3qn/
```

完整命令：

```powershell
$PY = "D:/Anaconda/envs/pytorch/python.exe"

& $PY Train_D3QN.py `
  --data_level low `
  --episodes 3000 `
  --output_root journal_results/v4-models/layer1/d3qn `
  --sequence_method iga `
  --outage_max_s 6 `
  --teacher mapguide `
  --guide_prob 0.5 `
  --guide_min_prob 0.5 `
  --guided_warmup_episodes 400 `
  --demo_fraction 0.25 `
  --demo_weight 1.0 `
  --demo_margin 0.8 `
  --bc_weight 1.0 `
  --bc_min_weight 0.25 `
  --bc_decay_episodes 600 `
  --n_step 10 `
  --action_repeat 2 `
  --pretrain_demo_episodes 40 `
  --pretrain_updates 20000 `
  --learning_rate 1e-4 `
  --epsilon_decay_steps 300000 `
  --seed 1 `
  --eval_interval 25
```

### 2.4b 层面 1 的连续 DRL 对比：PPO

DDPG 的旧正式运行在两种配方下均为 0/120 成功，且出现长时间停滞，保留为
内部负例，不进入新版主表。连续 DRL 对比改用 PPO。

固定条件：

- 低负载、6 s、Variant A、IGA 顺序、seed 1；
- 3000 episodes；
- PPO clipped surrogate + GAE；
- MapGuide 20 个示范回合预训练，随后 5000 次监督 actor updates；
- 网络宽度 256；
- actor learning rate `3e-4`，critic learning rate `1e-3`；
- `n_steps=1024`、batch 256、10 update epochs；
- gamma `0.995`、GAE lambda `0.95`、clip ratio `0.2`；
- 最终评估使用策略均值，不采样、不使用 MapGuide。

输出：

```text
journal_results/v4-models/ppo/A/low/ppo/
```

命令：

```powershell
& $PY Train_PPO.py `
  --data_level low `
  --episodes 3000 `
  --output_root journal_results/v4-models/ppo `
  --sequence_method iga `
  --outage_max_s 6 `
  --teacher mapguide `
  --pretrain_demo_episodes 20 `
  --pretrain_updates 5000 `
  --demo_noise_scale 0.25 `
  --net_width 256 `
  --learning_rate 3e-4 `
  --critic_learning_rate 1e-3 `
  --n_steps 1024 `
  --batch_size 256 `
  --update_epochs 10 `
  --gamma 0.995 `
  --gae_lambda 0.95 `
  --clip_ratio 0.2 `
  --entropy_coef 0.01 `
  --value_coef 0.5 `
  --max_grad_norm 0.5 `
  --seed 1 `
  --eval_interval 25
```

Win 端必须使用 GPU（`--device` 缺省会自动选择 CUDA）；如果显式指定设备，
可加 `--device cuda`。

### 2.5 层面 3 的三个 TD3 基线（Win）

全部在低负载、6 s 下重训：

- Distance-TSP+TD3，A2G-blind；
- EEGA-TD3，A2G-blind；
- Yang-style+TD3，A2G-blind；

输出：

```text
journal_results/v4-baselines/{distance-tsp,eega,yang}/
```

DRL-SA 与 SA-D3QN 为独立 D3QN 实验族，整体转移到 Mac，命令见 §3.6。
这样不拆分同一比较族，同时减少 Win 的 GPU 队列。

### 2.6 A2G 可见性消融

在低负载、6 s 下重训 Distance-TSP+TD3 的 A2G-aware 版本；A2G-blind 版本直接复用层面 3 的 Distance-TSP+TD3。

### 2.7 Win 回传门槛

每个 run 至少回传：

```text
run_config.json
eval_history.json
UAV_TD3_improved.npz
paper_figures/
training_results.jpg
preflight/
```

`run_config.json` 必须同时包含：

```json
{
  "state_schema": "v4-max-contiguous: ...",
  "qos_constraint": "max_contiguous_outage"
}
```

如字段缺失，Mac 拒收该结果。

## 3. Mac：确定性算法、两个 D3QN 基线与审计

Mac 负责确定性方法、两个 D3QN 盲基线，以及最终结果审计和图表：

### 3.1 A*

- low/medium/high 主基线，预算 6 s；
- QoS 预算 3/4/5/6/8 s；
- IGA 顺序与 TSP 顺序的顺序质量核查；
- 统一通过真实 World replay 产生 `T`、`τ`、`T_out^cont`、`T_out^total`。

当前已完成的主基线（6 s）：

| 档位 | A* |
|---|---:|
| low | 216.0 s |
| medium | 316.5 s |
| high | 445.5 s |

MapGuide 三档在 6 s 下均预检成功，`T_out^cont=2.5s`：

| 档位 | MapGuide T | 上传量 |
|---|---:|---:|
| low | 196.5 s | 450 MB |
| medium | 242.0 s | 900 MB |
| high | 305.0 s | 1500 MB |

IGA 顺序已复核：低负载 `[1,6,2,3,5,4]`，中、高负载 `[1,6,5,4,2,3]`。

低负载 6 s 的 A* 顺序审计：

| 顺序 | T+τ |
|---|---:|
| 距离 TSP `[1,6,5,2,3,4]` | 215.5 s |
| IGA `[1,6,2,3,5,4]` | 216.0 s |
| EEGA `[1,6,5,3,2,4]` | 222.5 s |
| Yang-style `[1,6,5,4,2,3]` | 223.5 s |

新版 A* 重放不再支持旧稿“IGA 顺序优于距离 TSP”的结论：当前差距为 IGA 慢 0.5 s。最终顺序贡献必须等待新 DRL 轨迹结果，在此之前不得沿用旧叙事。

low QoS 标定：

| 预算 | A* |
|---:|---:|
| 3 s | 不可行 |
| 4 s | 219.0 s |
| 5 s | 216.0 s |
| 6 s | 216.0 s |
| 8 s | 顺序分段搜索未找到路径，参照可留空 |

### 3.2 MapGuide

- 三档负载的 6 s 预检与最终 reference；
- QoS 3/4/5/6/8 s 的预检记录；
- 记录最长连续中断和累计中断。

### 3.3 PSO-trajectory

- 使用新版 World 重新优化低负载轨迹；
- 只能以真实 World replay 结果进入比较；
- 保留完整优化日志和目标收敛记录。

Mac 已在 6 s 下按原定 `swarm=40, iterations=60, seed=1` 重跑。PSO 在第 19
次迭代按连续 15 次无改进早停，最佳候选 `T_out^cont=6.5s`，因此任务失败。
该结果保留为正式失败记录，不增加预算或更换 seed。

### 3.4 顺序与公平性审计

- 重算各负载的 IGA 顺序；
- 核验所有 run 的地图哈希、payload、起终点、顺序和预算；
- 核验失败轨迹没有进入可行排名；
- 核验三权重在 6 s 的复用关系。

### 3.5 汇总与图表

Win 回传后由 Mac：

- 生成新版主结果表；
- 生成数据量、QoS、权重、A2G 消融图；
- 生成轨迹图与奖励曲线；
- 对比 v4 结果与 legacy，只做版本审计，不混入同一数值图；
- 更新论文写作手册中的最终数字。

### 3.6 DRL-SA 与 SA-D3QN（Mac / MPS）

这两个 D3QN 基线形成独立实验族，整体由 Mac 在 MPS 上执行，不拆分到
Win。两者均为低负载、6 s、5000 episodes、7 维 A2G-blind、MapGuide
G2A-only 教师、n-step 10、动作重复 2、DQfD 40 回合/20000 updates。

DRL-SA：

```bash
python3 Train_D3QN.py \
  --data_level low \
  --episodes 5000 \
  --output_root journal_results/v4-baselines/drl-sa \
  --sequence_method sa_service \
  --a2g_blind \
  --outage_max_s 6 \
  --teacher mapguide \
  --guide_prob 0.5 \
  --guide_min_prob 0.5 \
  --guided_warmup_episodes 400 \
  --demo_fraction 0.25 \
  --demo_weight 1.0 \
  --demo_margin 0.8 \
  --bc_weight 1.0 \
  --bc_min_weight 0.25 \
  --bc_decay_episodes 600 \
  --n_step 10 \
  --action_repeat 2 \
  --pretrain_demo_episodes 40 \
  --pretrain_updates 20000 \
  --learning_rate 1e-4 \
  --epsilon_decay_steps 300000 \
  --seed 1 \
  --eval_interval 25 \
  --device mps
```

SA-D3QN：

```bash
python3 Train_D3QN.py \
  --data_level low \
  --episodes 5000 \
  --output_root journal_results/v4-baselines/sa-d3qn \
  --sequence_method sa_distance \
  --a2g_blind \
  --outage_max_s 6 \
  --teacher mapguide \
  --guide_prob 0.5 \
  --guide_min_prob 0.5 \
  --guided_warmup_episodes 400 \
  --demo_fraction 0.25 \
  --demo_weight 1.0 \
  --demo_margin 0.8 \
  --bc_weight 1.0 \
  --bc_min_weight 0.25 \
  --bc_decay_episodes 600 \
  --n_step 10 \
  --action_repeat 2 \
  --pretrain_demo_episodes 40 \
  --pretrain_updates 20000 \
  --learning_rate 1e-4 \
  --epsilon_decay_steps 300000 \
  --seed 1 \
  --eval_interval 25 \
  --device mps
```

验收字段：

```text
state_dim = 7
a2g_blind = true
state_schema starts with v4-max-contiguous-blind
DRL-SA sequence_selector = sa_service
SA-D3QN sequence_selector = sa_distance
```

Mac 本地按顺序启动两项的脚本为 `run_mac_d3qn_baselines.sh`；先完成 DRL-SA，
再启动 SA-D3QN，避免两个 MPS 进程争用同一设备。

## 4. 推荐执行顺序

1. Mac 发布 v4 代码、测试、A* 标定和两份计划；
2. Win 只拉取 `feat/mac` 的 v4 版本，建立独立输出目录；
3. Win 先跑 3 s 冒烟预检和少量 episodes，确认 schema 与终止规则；
4. Mac 并行跑 A*、MapGuide、PSO 和顺序审计；
5. Win 跑 15 组 QoS；其中 6 s 三组同时供主方法和权重实验使用；
6. Win 跑 medium/high、层面 1 D3QN、三个 TD3 基线和 A2G 消融；
7. Mac 跑 DRL-SA/SA-D3QN，并汇总、核验和重画全部图表。

在 Win 完成 v4 冒烟验证前，不应启动长时间正式队列。

### Mac 当前进度

- [x] v4 环境、A*、预检、D3QN/TD3 评估口径
- [x] A* 三档 6 s 主参照
- [x] MapGuide 三档 6 s 预检
- [x] 三档 IGA 顺序复核
- [x] 低负载顺序质量 A* 审计
- [x] PSO 6 s 正式重跑并记录失败
- [~] Mac 正在跑 DRL-SA D3QN（2026-11-05 启动）
- [ ] Mac 等 DRL-SA 完成后跑 SA-D3QN D3QN
- [ ] Win 神经网络结果回传后的汇总、审计和图表

## 5. 验收原则

- 旧累计中断结果只作 legacy；
- 新结果必须显式记录 `max_contiguous_outage`；
- A* 和网络方法只比较真实的同一地图、顺序和约束；
- 不因某组失败而改 seed、改预算或换成旧模型；
- 所有最终表格、图和正文数字均从 v4 结果重新生成。
