# V5 Implementation Status

更新：2026-11-08

## 已完成

- `World` 支持 `qos_enforcement=hard|continue`；
- QoS 违约后继续任务；
- `mission_complete` 与 `qos_feasible` 独立记录；
- 1000 s 时域参数化；
- 模型选择支持完整但 QoS 违约的策略；
- 恢复旧 PRIGA 载荷/无线电边权；
- sequence-only GA、PSO、SA、A*；
- 顺序代理指标和 JSON/CSV 输出；
- DDPG、PPO、TD3、D3QN 的 V5 参数透传；
- SAC 训练器；
- MCO-A* 的 1000 s 和 relaxed completion fallback；
- Win 并行矩阵驱动 `run_v5_matrix.py`；
- 全量测试通过。

## Mac 已完成结果

### MCO-A*

1000 s, `T_out_cont<=6 s`, `qos_enforcement=continue`：

| Level | T | tau | T+tau | T_out_cont | Constraint |
|---|---:|---:|---:|---:|---|
| Low | 287.5 | 170.0 | 457.5 | 6.0 | pass |
| Medium | 459.5 | 329.5 | 789.0 | 5.5 | pass |
| High | 663.0 | 533.0 | 1196.0 | 5.5 | pass |

结果目录：

```text
journal_results/v5-deterministic/mco-astar/
```

### Dual-map QoS-constrained A*

E1 uses the repository's dual-map A* rather than MCO-A*. Its state transition
uses the A2G uplink rate to determine payload reduction and upload delay, and
the G2A outage map to track and prune the maximum contiguous outage. It keeps
the PRIGA order fixed and solves each leg sequentially, so the result is a
feasible trajectory baseline rather than a joint-order optimum.

1000 s, `T_out_cont<=6 s`, `N=2000` slots:

| Level | T | tau | T+tau | T_out_cont | Constraint |
|---|---:|---:|---:|---:|---|
| Low | 138.0 | 78.0 | 216.0 | 5.0 | pass |
| Medium | 189.5 | 127.0 | 316.5 | 4.0 | pass |
| High | 254.5 | 191.0 | 445.5 | 4.0 | pass |

结果目录：

```text
journal_results/v5-deterministic/dual-map-astar/
```

MCO-A* remains in E5 as the adapted, G2A-only literature baseline. It must not
be relabelled as the dual-map A* used in E1.

### Sequence proxy

七种顺序方法的三档结果：

```text
journal_results/v5-sequence/sequence-study.json
journal_results/v5-sequence/sequence-study.csv
```

Low：

| Method | Order | Proxy T+tau | Payload-weighted A2G rate | Mean leg outage |
|---|---|---:|---:|---:|
| PRIGA (original) | 1-6-2-3-5-4 | 252.9 | 3.116 | 3.92 |
| Yang-style | 1-6-5-4-2-3 | 248.5 | 3.414 | 3.29 |
| Standard GA | 1-6-5-2-3-4 | 257.1 | 2.750 | 3.48 |
| PSO | 1-6-5-2-3-4 | 257.1 | 2.750 | 3.48 |
| SA | 1-6-5-2-3-4 | 257.1 | 2.750 | 3.48 |
| A* | 1-6-5-2-3-4 | 257.1 | 2.750 | 3.48 |
| EEGA-style | 1-6-5-3-2-4 | 264.1 | 2.780 | 3.36 |

Medium 和 High 中旧 PRIGA 均取得最低代理目标值；Low 低于 Yang-style。
该代理不是 World replay，最终结论仍由 CA-TD3 端到端实验决定。

## Win 启动

先停止所有 legacy-v4 任务。拉取本分支后：

```bash
python run_v5_matrix.py \
  --groups E1 E2 E4 \
  --levels low medium high \
  --device cuda \
  --workers 3 \
  --execute
```

建议先做 Low gate：

```bash
python run_v5_matrix.py \
  --groups E1 E2 \
  --levels low \
  --device cuda \
  --workers 3 \
  --execute
```

Low gate 通过后，Win 启动 E1/E2/E4 的 Medium/High：

```bash
python run_v5_matrix.py \
  --groups E1 E2 E4 \
  --levels medium high \
  --device cuda \
  --workers 3 \
  --execute
```

最后补 E4 Low：

```bash
python run_v5_matrix.py \
  --groups E4 \
  --levels low \
  --device cuda \
  --workers 3 \
  --execute
```

Mac 负责：

```bash
python run_v5_matrix.py \
  --groups E5 \
  --levels low medium high \
  --device mps \
  --workers 4 \
  --skip-shared-proposed \
  --output-root journal_results/v5-mac \
  --execute
```

E3 的 Medium/High 四组由 Mac 的 MPS 并行执行，E5 的其余 15 个基线也由
Mac 重新训练。E5 的 PRIGA+CA-TD3 复用 Win 的 E1 结果，不重复训练。

输出到：

```text
journal_results/v5-mac/E3/
journal_results/v5-mac/E5/
```

因此 Win 的正式队列是 51 个唯一训练格。Mac 承担 E3 的 4 个训练格和
E5 的 15 个基线训练格。

Mac 当前承担的完整实验块：

- Dual-map A*（E1）：Low/Medium/High，共 3 组，已完成；
- MCO-A*（E5）：Low/Medium/High，共 3 组，已完成；
- sequence-only proxy：7 种顺序 x 3 档，共 21 个确定性评估，已完成；
- E3 A2G ablation：Medium/High x blind/visible，共 4 组，正在运行；
- E5 literature baselines：6 种方法 x 3 档，共 15 组，复用 Win Proposed。

E1/E2 整体放在 Win：

- E1 需要五种轨迹算法同机比较；
- E2 需要七种顺序对应的 CA-TD3 训练同机完成；

## 当前风险

- V5 是新的 1000 s 协议，不能把旧 500 s 训练结果混入主表；
- PRIGA 在 Low 的代理目标与 Yang-style 并列，尚未证明严格领先；
- SAC 已通过单元和短 smoke test，但尚未在 Win 上进行正式长训练；
- DDPG/PPO/SAC 的正式可行性必须由 Win 的 CUDA pilot 决定；
- 当前所有训练均为 seed 1，暂无 seed-level uncertainty。
