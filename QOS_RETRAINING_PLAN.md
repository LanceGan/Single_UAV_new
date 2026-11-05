# QoS 逐预算重训实验方案

> 交接对象：Win 实验端
> 制定日期：2026-11-05
> 数据来源分支：`feat/mac`
> 实验性质：低负载下，固定目标权重，但每个 QoS 预算独立训练一个 TD3

## 1. 实验目的

旧 QoS 扫描使用 `30/40/60/80/100 s`。由于实际 `T_out` 通常只有 15–28 s，40 s 以上的预算基本不再束缚可行解，不能形成有效的五点敏感性曲线。

本实验重新扫描 `20/25/30/35/40 s`，并对每个预算重新训练，回答：

1. QoS 预算从不可行边缘收紧到非束缚区时，三个目标权重下的策略目标值如何变化；
2. 更紧的 QoS 约束是否以更长任务时间和更高上传时延为代价；
3. 任务时间优先与传输时延优先两类策略对 QoS 收紧的鲁棒性是否不同。

本实验取代当前 `journal_results/qos-sweep-reweighted/` 的免重训后评估图。后者保留为中间对照，不能与逐预算重训结果混在同一正式曲线中。

## 2. 实验矩阵

采用低负载、完全相同场景，共训练：

```text
3 个目标权重 × 5 个 QoS 预算 = 15 个完整 TD3 运行
```

| 配置名 | `w1`（T） | `w2`（τ） | QoS 预算 |
|---|---:|---:|---|
| `w80-20` | 0.8 | 0.2 | 20, 25, 30, 35, 40 s |
| `w50-50` | 0.5 | 0.5 | 20, 25, 30, 35, 40 s |
| `w20-80` | 0.2 | 0.8 | 20, 25, 30, 35, 40 s |

不要把五个预算合并训练，也不要用 100 s 模型替代。每个预算必须从头训练。

## 3. 固定实验条件

所有 15 次运行必须一致：

| 参数 | 固定值 |
|---|---|
| 数据档位 | `low` |
| 数据向量 | `[90,90,60,60,90,60] MB` |
| 巡检顺序 | Data-aware IGA，预期 `[1,6,2,3,5,4]` |
| 轨迹算法 | TD3 |
| 算法参数名 | `td3_bc`，论文显示为 TD3 |
| 状态/动作变体 | `A` |
| 示范教师 | MapGuide |
| 示范初始概率 | 0.5 |
| 示范概率下限 | 0.2 |
| 示范衰减 | 400 episodes |
| BC 权重 | 1.0 → 0.25，600 episodes |
| 示范 batch 比例 | 0.25 |
| 训练回合 | 3000 |
| 评估间隔 | 25 episodes |
| seed | 1 |
| 学习率 | actor `1e-4`，critic `1e-3` |
| 网络宽度 | 256 |
| replay capacity | 200000 |
| gamma | 0.995 |
| 最大速度 | 50 m/s |
| 时隙 | 0.5 s |
| 回合长度 | 1000 slots / 500 s |
| 地图 | 固定 G2A + A2G 地图 |

每个预算的状态归一化、训练终止和正式评估都使用该预算自己的 `outage_max_s`。不要传 `state_outage_max_s` 覆盖值。

## 4. 预算选择依据

2026-11-05 已用 MapGuide 在低负载、IGA 顺序下完成预检：

| 预算 | 预检 | MapGuide T | MapGuide T_out |
|---:|---|---:|---:|
| 20 s | 成功 | 196.5 s | 14.5 s |
| 25 s | 成功 | 196.5 s | 14.5 s |
| 30 s | 成功 | 196.5 s | 14.5 s |

因此 20 s 预算下训练前预检可以通过，不会因 MapGuide 教师路径超限而无法启动。

A* 顺序分段参照已重跑：

| 预算 | A* 状态 | T+τ | 实际 T_out |
|---:|---|---:|---:|
| 20 s | 未找到可行路径 | 无 | — |
| 25 s | 可行 | 235.0 s | 25.0 s |
| 30 s | 可行 | 214.0 s | 27.5 s |
| 35 s | 可行 | 214.0 s | 27.5 s |
| 40 s | 可行 | 214.0 s | 27.5 s |

A* 是顺序分段搜索，不证明全局最优；20 s 失败只表示当前 25 m 栅格和搜索模式下未找到路径。

## 5. 输出目录

每组输出统一放在：

```text
journal_results/qos-retrain/<weight>/<budget>/A/low/td3_bc/
```

例如：

```text
journal_results/qos-retrain/w80-20/qos20/A/low/td3_bc/
journal_results/qos-retrain/w50-50/qos30/A/low/td3_bc/
journal_results/qos-retrain/w20-80/qos40/A/low/td3_bc/
```

不得覆盖现有：

- `journal_results/qos-sweep/`
- `journal_results/qos-sweep-reweighted/`
- `journal_results/weight-sweep/`

## 6. Win 执行命令

以下为 PowerShell 批量命令。先修改 `$PY` 为本机可用的 Python 3.11+ 环境；该环境必须能导入 Torch、NumPy、Matplotlib。

```powershell
$PY = "D:/Anaconda/envs/pytorch/python.exe"

$weights = @(
    [pscustomobject]@{ Name = "w80-20"; W1 = 0.8; W2 = 0.2 },
    [pscustomobject]@{ Name = "w50-50"; W1 = 0.5; W2 = 0.5 },
    [pscustomobject]@{ Name = "w20-80"; W1 = 0.2; W2 = 0.8 }
)
$budgets = 20, 25, 30, 35, 40

foreach ($weight in $weights) {
    foreach ($budget in $budgets) {
        $RunRoot = "journal_results/qos-retrain/$($weight.Name)/qos$budget"

        & $PY Train_SINR.py `
            --variant A `
            --data_level low `
            --algorithm td3_bc `
            --teacher mapguide `
            --outage_max_s $budget `
            --objective_w1 $weight.W1 `
            --objective_w2 $weight.W2 `
            --episodes 3000 `
            --seed 1 `
            --eval_interval 25 `
            --guided_warmup_episodes 400 `
            --guide_prob 0.5 `
            --guide_min_prob 0.2 `
            --bc_weight 1.0 `
            --bc_min_weight 0.25 `
            --bc_decay_episodes 600 `
            --demo_fraction 0.25 `
            --output_root $RunRoot

        if ($LASTEXITCODE -ne 0) {
            throw "QoS run failed: $($weight.Name) / $budget s"
        }
    }
}
```

每个长期任务建议单独保存终端日志，例如：

```powershell
$RunRoot = "journal_results/qos-retrain/w80-20/qos20"
& $PY Train_SINR.py ... --output_root $RunRoot 2>&1 |
    Tee-Object -FilePath "$RunRoot.console.txt"
```

## 7. 启动前冒烟检查

先执行环境与预检：

```powershell
& $PY -c "import torch, numpy, matplotlib; print(torch.__version__, torch.cuda.is_available())"

& $PY Train_SINR.py `
  --variant A --data_level low --algorithm td3_bc --teacher mapguide `
  --outage_max_s 20 --objective_w1 0.8 --objective_w2 0.2 `
  --seed 1 --output_root journal_results/qos-retrain-smoke `
  --preflight_only
```

通过标准：

- 输出 `Mission preflight: ... 'success': True`；
- `outage_s` 不高于 20 s；
- `uploaded_mb = 450`；
- `visited_points = 6`。

## 8. 单次运行验收

每次训练完成后检查：

1. `run_config.json` 中 `objective_w1/objective_w2/outage_max_s` 与实验矩阵一致；
2. `payloads_mb` 为 `[90,90,60,60,90,60]`；
3. `order` 为 IGA 输出，三权重和五预算之间应一致；
4. 两个地图哈希一致；
5. `eval_history.json` 包含 120 次周期评估；
6. `eval_best.success = true`；
7. `eval_best.outage_time_s <= outage_max_s`；
8. 不从训练奖励曲线挑选结果，只使用纯 Actor、无噪声、无 MapGuide 控制的 `eval_best`。

如某次运行没有可行 `eval_best`，必须保留失败记录，不能换 seed、延长回合或修改权重后冒充原配置。

## 9. 最终汇总字段

每个权重、每个预算记录：

| 字段 | 定义 |
|---|---|
| `budget_s` | QoS 预算 |
| `objective_s` | `2(w1*T + w2*tau)` |
| `task_time_s` | `T` |
| `transmission_delay_s` | `tau_total` |
| `outage_time_s` | 实际 `T_out` |
| `success_rate` | 120 次周期评估中成功比例 |
| `best_episode` | `eval_best` 对应 episode |
| `termination_reason` | 应为 `success` |
| `feasible` | `eval_best.success` |

推荐最终图包含：

1. 三个权重各自的加权目标 vs QoS 预算；
2. 三个权重的 `T` vs QoS 预算；
3. 三个权重的 `τ_total` vs QoS 预算；
4. 实际 `T_out` 与可行评估率。

三个权重使用不同目标，不能直接按 `J` 的绝对数值跨权重排名；应比较各自随预算的变化及 `T/τ` 分解。

## 10. 预期趋势与判定

对每个固定权重，预算扩大时可行集只扩大，因此理论最优目标应满足：

```text
J*(20s) >= J*(25s) >= J*(30s) >= J*(35s) >= J*(40s)
```

有限训练和单 seed 允许小幅波动，但应观察到：

- 20–25 s：约束明显生效，目标值下降较快，可行率上升；
- 25–30 s：从强约束向弱约束过渡；
- 35–40 s：目标值和可行率接近平台；
- 偏 `T` 权重维持较短 `T`、较高 `τ`；
- 偏 `τ` 权重维持较低 `τ`、较长 `T`；
- 实际 `T_out` 始终不超过预算，且在平台区接近策略自然中断水平。

若 25 s 的结果比 30 s 更好，或 35/40 s 仍有系统性大幅改善，应检查：

- `run_config.json` 是否配置错位；
- 三个权重是否使用了同一 IGA 顺序；
- `eval_best` 是否按正确权重选择；
- 是否把失败回合或提前终止值混入目标曲线；
- 是否存在明显的单 seed 训练波动。

## 11. Win 回传清单

15 组每组至少回传：

```text
run_config.json
eval_history.json
UAV_TD3_improved.npz
paper_figures/
training_results.jpg
preflight/
```

建议额外提供每组 `models/*_actoreval_best.pth` 的外部压缩包，用于结果复核和轨迹重渲染；不要把所有中间 actor/critic checkpoint 或 TensorBoard 原始日志推入 Git。

回传时创建一个汇总：

```text
journal_results/qos-retrain-summary.json
```

包含 15 组的 `budget/weight/T/tau/T_out/objective/success_rate/best_episode`，并注明：

```text
protocol = retrained independently at each QoS budget
seed = 1
```

## 12. 完成定义

以下条件全部满足才算实验完成：

- 15 个正式 run 全部有 `run_config.json` 和 `eval_history.json`；
- 所有完成运行通过 map hash、payload、order 和预算配置一致性检查；
- 每个成功 run 的 `eval_best` 实际 `T_out` 不超过对应预算；
- 三个权重各有五个有效预算点；
- 生成目标、`T`、`τ`、`T_out`、可行率曲线；
- 失败 run 有明确记录，未用其他配置替代；
- 明确区分“逐预算重训曲线”和旧的“固定策略后评估曲线”。
