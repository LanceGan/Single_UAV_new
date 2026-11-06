# v4 eval-best actor bundle（Proposed 轨迹重画用）

> 生成日期：2026-11-06
> 用途：Mac 侧用 `evaluate_policy.py` 在 CPU 上无噪声、无 MapGuide 重放 `eval_best` 轨迹，重画双地图轨迹图。

## 训练代码版本

- 训练 worktree：`F:\Codes\Python\Single_UAV_mac`，分支 `feat/mac`。
- QoS 15 组 TD3 的 worker 在 commit `647a414` 启动；训练过程中 worktree 快进到 `4010095`、`dc4f162`。
- 该窗口内对 TD3 的改动仅为设备抽象（`TD3.py`/`DDPG.py` 增加 `device` 参数、`Train_SINR.py` 增加 `--device`），**算法语义不变**。
- medium / high / PPO 在 commit `dc4f162` 启动；`Train_PPO.py` 首次引入于 `4010095`。
- 统一口径：`state_schema = v4-max-contiguous`，`qos_constraint = max_contiguous_outage`。

## 文件清单

| 文件 | 来源 run | eval_best episode | success |
|---|---|---|---|
| `w80-20_qos6_actor.pth` | `v4-qos/w80-20/qos6` | 1275 | true |
| `w50-50_qos6_actor.pth` | `v4-qos/w50-50/qos6` | 1550 | true |
| `w20-80_qos6_actor.pth` | `v4-qos/w20-80/qos6` | 1800 | true |
| `medium_qos6_actor.pth` | `v4-models/load/medium` | 2100 | true |
| `high_qos6_actor.pth` | `v4-models/load/high` | 1725 | true |
| `ppo_low_qos6_actor.pth` | `v4-models/ppo`（失败负例） | 25 | false |

SHA-256 与 source_run / episode 的对应关系见 `MANIFEST.json`。

## 说明

- PPO checkpoint 为失败负例（0/120 成功，`T_out^cont=6.5s > 6s`），非正式主结果，仅作可选失败轨迹同步。
- 未打包 critic、period checkpoint、TensorBoard 日志；轨迹重放只需要 actor。
