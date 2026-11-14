# Win → Mac 同步请求：V5 E1 Medium/High eval-best 轨迹

- 请求方：Mac
- 执行方：Win
- 目标分支：`feat/win`
- 当前 Mac 基线：`feat/mac` 最新提交，已 fetch 的 Win 基线为 `5517943`

## 1. 为什么需要这批文件

Mac 已有 D3QN、PPO、SAC 的 Medium/High 训练历史、
`eval_history.json`、`run_config.json` 和最终汇总指标，但没有下列六个
`eval-best` actor，因此无法重放论文所需的真实轨迹：

```text
journal_results/v5-win/E1/d3qn/medium/d3qn/models/d3qn_eval_best.pt
journal_results/v5-win/E1/d3qn/high/d3qn/models/d3qn_eval_best.pt
journal_results/v5-win/E1/ppo/A/medium/ppo/models/ppo_ppo_actoreval_best.pth
journal_results/v5-win/E1/ppo/A/high/ppo/models/ppo_ppo_actoreval_best.pth
journal_results/v5-win/E1/sac/A/medium/sac/models/sac_sac_actoreval_best.pth
journal_results/v5-win/E1/sac/A/high/sac/models/sac_sac_actoreval_best.pth
```

不要用 preflight、MapGuide、最终 actor、其他 episode actor 或插值轨迹代替。
这些文件只用于 E1 的 A2G/G2A 轨迹图；不需要重训，也不需要重推已有的奖励
曲线和可行率曲线。

## 2. 请优先推送的固定目录

请不要把文件放回被 `.gitignore` 忽略的 `journal_results/**/models/`，而是在
`feat/win` 分支建立以下同步包：

```text
sync_bundles/v5_e1_medium_high_actors/
├── MANIFEST.json
├── d3qn_medium_actor_eval_best.pt
├── d3qn_high_actor_eval_best.pt
├── ppo_medium_actor_eval_best.pth
├── ppo_high_actor_eval_best.pth
├── sac_medium_actor_eval_best.pth
├── sac_high_actor_eval_best.pth
├── trajectory/
│   ├── d3qn_medium_trajectory.npz
│   ├── d3qn_high_trajectory.npz
│   ├── ppo_medium_trajectory.npz
│   ├── ppo_high_trajectory.npz
│   ├── sac_medium_trajectory.npz
│   └── sac_high_trajectory.npz
└── metrics/
    ├── d3qn_medium_metrics.json
    ├── d3qn_high_metrics.json
    ├── ppo_medium_metrics.json
    ├── ppo_high_metrics.json
    ├── sac_medium_metrics.json
    └── sac_high_metrics.json
```

actor 文件是必需项。原 Win CUDA 环境下生成的轨迹 NPZ 和 metrics 也强烈建议
一并提供，因为 Mac 与 Win 的设备数值精度不同，跨设备重放可能产生不同分支。

## 3. 六个 run 的来源和验收指标

| 方法 | 档位 | actor 原始路径 | selector | episode | T (s) | tau (s) | T+tau (s) | T_out^cont (s) | 巡检/上传 | 状态 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| D3QN | Medium | `journal_results/v5-win/E1/d3qn/medium/d3qn/models/d3qn_eval_best.pt` | `best` | 950 | 248.5 | 153.0 | 401.5 | 4.5 | 6/6, 900/900 MB | 可行 |
| D3QN | High | `journal_results/v5-win/E1/d3qn/high/d3qn/models/d3qn_eval_best.pt` | `best` | 1275 | 319.5 | 213.0 | 532.5 | 3.0 | 6/6, 1500/1500 MB | 可行 |
| PPO | Medium | `journal_results/v5-win/E1/ppo/A/medium/ppo/models/ppo_ppo_actoreval_best.pth` | `eval_best` | 2950 | 732.0 | 212.0 | 944.0 | 30.0 | 6/6, 900/900 MB | 完成但违反 QoS |
| PPO | High | `journal_results/v5-win/E1/ppo/A/high/ppo/models/ppo_ppo_actoreval_best.pth` | `eval_best` | 1975 | 643.0 | 298.5 | 941.5 | 17.0 | 6/6, 1500/1500 MB | 完成但违反 QoS |
| SAC | Medium | `journal_results/v5-win/E1/sac/A/medium/sac/models/sac_sac_actoreval_best.pth` | `eval_best` | 1925 | 1000.0 | 237.5 | 1237.5 | 12.0 | 6/6, 900/900 MB | 完成但违反 QoS |
| SAC | High | `journal_results/v5-win/E1/sac/A/high/sac/models/sac_sac_actoreval_best.pth` | `eval_best` | 1100 | 459.5 | 290.5 | 750.0 | 4.0 | 6/6, 1500/1500 MB | 可行 |

注意：

- D3QN 的 `eval_history.json` 中 `eval_best` 为 `null`，有效选择记录在
  `best` 字段，因此必须使用 episode 950 和 1275。
- 上述状态以现有 `eval_history.json` 为准。完整但违反 QoS 的结果仍要保留，
  不能因为不可行就不推送。

## 4. 这些 run 的共同协议

六次重放必须保持原配置，不得为了“让曲线好看”重新训练或改参数：

- seed：`1`
- 控制：确定性 actor/eval policy，无探索噪声、无 MapGuide
- 时隙长度：`0.5 s`
- horizon：`total_steps=2000`
- QoS：`max_contiguous_outage`，预算 `6 s`
- outage probability threshold：`0.5`
- A2G：可见，`a2g_blind=false`
- state dimension：`8`
- net width：`256`
- sequence：PRIGA，顺序 `[1,6,5,4,2,3]`
- Medium payload：`[180,180,180,120,120,120] MB`
- High payload：`[210,290,290,290,210,210] MB`
- 跨设备重放时使用该 run 的 `run_config.json` 记录的设备；
  当前六组均记录为 `cuda`。

## 5. 轨迹 NPZ 的推荐格式

每个 `*_trajectory.npz` 对应一次完整的 eval-best deterministic rollout，
数组单位为：

- 位置：km
- 时间、中断：s
- payload：MB

至少包含：

```text
positions                         # (N, 2), 单位 km
time_s                            # (N,), 0, 0.5, 1.0, ...
pending_mb                        # (N,)
visited_points                    # (N,), 已巡检点数量
outage_s                          # (N,), 该 rollout 截至当前的最大连续中断
total_outage_s                    # (N,), 累计中断时长
current_contiguous_outage_s       # (N,), 当前连续中断；进入正常时隙立即归零
actions                           # (N, action_dim), 可选但建议提供
```

如有以下数组也请保留：

```text
a2g_snr_db
g2a_outage_probability
```

硬性一致性要求：

```text
len(positions) == len(time_s) == len(pending_mb) == len(visited_points)
```

并且：

- `positions` 全部有限；
- `time_s` 单调不减；
- 最终 `visited_points == 6`；
- 最终 `pending_mb == 0`；
- `max(current_contiguous_outage_s)` 必须等于该 run 的
  `T_out^cont`；
- 最终 `time_s`、`T+tau`、巡检数和上传量必须与 `eval_history.json`
  selector 记录一致。

不要提交 preflight trace，也不要把累计中断写进
`current_contiguous_outage_s`。

## 6. metrics JSON 至少包含

每个 `metrics/*.json` 请至少包含以下字段：

```json
{
  "algorithm": "d3qn",
  "level": "medium",
  "source_commit": "<Win commit>",
  "source_run": "journal_results/v5-win/E1/d3qn/medium/d3qn",
  "eval_history_selector": "best",
  "eval_best_episode": 950,
  "actor_path": "sync_bundles/v5_e1_medium_high_actors/d3qn_medium_actor_eval_best.pt",
  "actor_sha256": "<sha256>",
  "trajectory_path": "sync_bundles/v5_e1_medium_high_actors/trajectory/d3qn_medium_trajectory.npz",
  "trajectory_sha256": "<sha256>",
  "replay_device": "cuda",
  "torch_version": "<version>",
  "cuda_version": "<version>",
  "task_time_s": 248.5,
  "transmission_delay_s": 153.0,
  "objective_s": 401.5,
  "contiguous_outage_s": 4.5,
  "total_outage_s": "<actual value>",
  "visited_points": 6,
  "uploaded_mb": 900.0,
  "mission_complete": true,
  "qos_feasible": true,
  "matches_eval_history": true
}
```

`MANIFEST.json` 请按同样内容汇总六项，并额外记录：

- 生成/重放时间；
- 生成脚本或命令；
- 地图文件 hash；
- 是否在 Win 原设备完成重放；
- 哪一项若无法提供，以及原因。

## 7. Win 推送前自检

请在 Win 仓库根目录执行并确认：

1. 六个 actor 均能被对应算法正常加载。
2. 每个 actor 的 deterministic replay 与 `eval_history.json` 的 selector
   指标逐项一致。
3. 每个 NPZ 长度一致、数组有限、时间单调。
4. 最终巡检为 6/6，最终上传完成。
5. `current_contiguous_outage_s` 的最大值与 `T_out^cont` 一致。
6. actor、trajectory、metrics 的 SHA-256 已写入 MANIFEST。

如果 actor 可以找到但跨设备轨迹无法逐值复现，请不要删除或修改原记录；仍提交
actor、原始 metrics 和 Win 本机轨迹，并在 MANIFEST 中写明
`matches_eval_history=false` 及具体差异。

## 8. Git 操作

同步包不在 `.gitignore` 的 `models/` 路径下，因此不需要 `git add -f`：

```bash
git checkout feat/win
git pull --ff-only
git add sync_bundles/v5_e1_medium_high_actors
git commit -m "results: sync V5 E1 medium/high eval-best actors and trajectories"
git push new feat/win
```

如果 Win 的远端名不是 `new`，请使用其本仓库实际远端名。完成后请回复：

- commit hash；
- 六个 actor 的 SHA-256；
- 六个 trajectory 的 SHA-256；
- 哪些项与 `eval_history.json` 完全一致；
- 任何不满足项及原因。

## 9. Mac 收到后的处理

Mac 将：

1. 逐项核对 Git commit、actor SHA-256、trajectory SHA-256 和 metrics；
2. 优先使用 Win 原设备轨迹，避免跨设备分支差异；
3. 生成 E1 的 A2G SNR 和 G2A coverage-probability 共 8 张独立面板；
4. 每张面板叠加 Low/Medium/High，包含起点、终点、巡检点、完整图例和色条；
5. 完成 PDF 字号、碰撞、面板布局和论文编译检查。

除非 Mac 验证发现不一致，Win 不需要重训、不需要重画论文图，也不需要修改
论文仓库。
