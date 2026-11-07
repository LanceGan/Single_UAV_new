# Win Optional DRL Matrix

> 日期：2026-11-07
> 范围：除多 seed 外，把原“可选”实验全部补齐。所有神经网络训练交给 Win；
> Mac 负责确定性顺序推导、Preflight、结果接收、审计和画图。

## 1. 运行总量

Win 新增 14 组正式训练：

| 实验族 | Medium | High | 小计 |
|---|---:|---:|---:|
| Layer-1 D3QN | 1 | 1 | 2 |
| PPO | 1 | 1 | 2 |
| Distance-TSP / EEGA / Yang-style + TD3 | 3 | 3 | 6 |
| DRL-SA / SA-D3QN | 2 | 2 | 4 |
| **合计** | **7** | **7** | **14** |

多 seed 不在本轮范围。

Outage-probability threshold robustness (`kappa_out = 0.5, 0.3, 0.2`) is also
deferred and is not part of these 14 runs. See `DEFERRED_EXPERIMENTS.md`.

## 2. 已冻结的 seed-1 顺序

Win 训练前必须核对实际顺序与下表一致。若代码重新搜索产生不同顺序，停止该
run 并回传日志，不得把新顺序冒充本任务单中的正式顺序。

| 方法 | Low | Medium | High |
|---|---|---|---|
| PRIGA | `[1,6,2,3,5,4]` | `[1,6,5,4,2,3]` | `[1,6,5,4,2,3]` |
| Distance-TSP / SA-D3QN | `[1,6,5,2,3,4]` | `[1,6,5,2,3,4]` | `[1,6,5,2,3,4]` |
| EEGA | `[1,6,5,3,2,4]` | `[1,6,5,2,3,4]` | `[1,6,5,2,3,4]` |
| Yang-style / DRL-SA | `[1,6,5,4,2,3]` | `[1,6,5,3,2,4]` | `[2,3,5,6,1,4]` |

注意：

- SA-D3QN 与 Distance-TSP 的顺序相同，但轨迹算法不同，因此两组都要训练。
- DRL-SA 与 Yang-style 在 Medium/High 的顺序相同，但轨迹算法和信息集不同，
  因此两组都要训练。

## 3. Layer-1 D3QN：Medium/High

条件：

- A2G-aware；
- PRIGA 顺序；
- 3000 episodes，seed 1；
- 其余配方与现有 Low D3QN 完全一致。

输出：

```text
journal_results/v4-models/layer1/d3qn/medium/d3qn
journal_results/v4-models/layer1/d3qn/high/d3qn
```

命令模板，将 `<LEVEL>` 替换为 `medium` 或 `high`：

```powershell
python Train_D3QN.py `
  --data_level <LEVEL> `
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
  --eval_interval 25 `
  --device cuda
```

## 4. PPO：Medium/High

条件：

- A2G-aware；
- PRIGA 顺序；
- 3000 episodes，seed 1；
- PPO 配方与现有 Low PPO 完全一致。

输出：

```text
journal_results/v4-models/ppo/medium/ppo
journal_results/v4-models/ppo/high/ppo
```

命令模板：

```powershell
python Train_PPO.py `
  --data_level <LEVEL> `
  --episodes 3000 `
  --output_root journal_results/v4-models/ppo `
  --sequence_method iga `
  --outage_max_s 6 `
  --variant A `
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
  --eval_interval 25 `
  --device cuda
```

## 5. Distance-TSP / EEGA / Yang-style + TD3：Medium/High

这一族是 A2G-unaware 的完整文献基线，不是 PRIGA 顺序消融。

条件：

- A2G-blind；
- 3000 episodes，seed 1；
- 相同的 TD3、MapGuide G2A-only 教师和训练协议。

输出：

```text
journal_results/v4-baselines/{distance-tsp,eega,yang}/<LEVEL>/td3_bc
```

公共命令模板：

```powershell
python Train_SINR.py `
  --variant A `
  --data_level <LEVEL> `
  --algorithm td3_bc `
  --teacher mapguide `
  --guide_prob 0.5 `
  --guide_min_prob 0.2 `
  --guided_warmup_episodes 400 `
  --bc_weight 1.0 `
  --bc_min_weight 0.25 `
  --bc_decay_episodes 600 `
  --demo_fraction 0.25 `
  --episodes 3000 `
  --seed 1 `
  --eval_interval 25 `
  --outage_max_s 6 `
  --objective_w1 0.5 `
  --objective_w2 0.5 `
  --a2g_blind `
  --device cuda `
  --sequence_method <SELECTOR> `
  --output_root <OUTPUT_ROOT>
```

替换：

| Method | SELECTOR | OUTPUT_ROOT |
|---|---|---|
| Distance-TSP | `distance_tsp` | `journal_results/v4-baselines/distance-tsp` |
| EEGA | `eega` | `journal_results/v4-baselines/eega` |
| Yang-style | `yang` | `journal_results/v4-baselines/yang` |

## 6. DRL-SA / SA-D3QN：Medium/High

条件：

- A2G-blind；
- 7 维 A2G-blind 状态；
- 5000 episodes，seed 1；
- D3QN 配方与 Low 完全一致。

输出：

```text
journal_results/v4-baselines/{drl-sa,sa-d3qn}/<LEVEL>/d3qn
```

公共命令模板：

```powershell
python Train_D3QN.py `
  --data_level <LEVEL> `
  --episodes 5000 `
  --output_root <OUTPUT_ROOT> `
  --sequence_method <SELECTOR> `
  --a2g_blind `
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
  --eval_interval 25 `
  --device cuda
```

替换：

| Method | SELECTOR | OUTPUT_ROOT |
|---|---|---|
| DRL-SA | `sa_service` | `journal_results/v4-baselines/drl-sa` |
| SA-D3QN | `sa_distance` | `journal_results/v4-baselines/sa-d3qn` |

## 7. 验收门槛

每个 run 必须记录：

- `state_schema` 以 `v4-max-contiguous` 开头；
- `qos_constraint = max_contiguous_outage`；
- payload 与对应负载一致；
- 实际巡检顺序与第 2 节冻结表一致；
- 3000 或 5000 episodes、seed 1、每 25 episodes 评估；
- `eval_best` 为纯 actor、无教师、无探索噪声；
- 失败结果保留，不换 seed、不扩大预算、不延长回合。

每组至少回传：

```text
run_config.json
eval_history.json
UAV_TD3_improved.npz       # TD3/PPO 族
training_history.npz       # D3QN 族
preflight/
paper_figures/
training_results.jpg
```

## 8. Mac 侧负责

- 冻结并核验全部顺序；
- 为每个顺序生成 MapGuide 预检；
- 接收 Win 结果；
- 检查 schema、地图哈希、A2G 标志、episodes 和 seed；
- 统一重放 actor、生成表格和图表；
- 不把不可行 `eval_best` 放入排名。

Mac 已完成首轮冻结和预检：

```text
journal_results/v4-sequence-orders/orders.json
journal_results/v4-sequence-preflight/summary.json
```

三档所有唯一顺序在当前 MapGuide 参考轨迹下均完成 6/6 巡检和完整上传，
因此不存在“顺序本身无法形成可行参考路径”的阻断。该预检只验证环境与顺序
可行性，不替代 Win 的正式 DRL 训练结果。
