# Win 正式顺序消融任务单

> 日期：2026-11-07
> 基线分支：`feat/mac`
> 任务目的：在同一个 A2G-aware TD3 轨迹学习器和完全相同的训练协议下，只替换巡检顺序，单独检验 IGA 顺序是否给 Proposed 带来增益。

## 1. 要回答的问题

正式问题是：

> 固定标准 TD3、双地图可见性、状态、动作、奖励、QoS、训练回合和评价协议，只改变巡检顺序，IGA 是否产生独立且可重复的目标值改善？

这组实验与现有层面 3 不同。现有层面 3 的基线和 Proposed 同时改变了
“顺序算法”和“轨迹栈是否读取 A2G 地图”，因此不能单独证明 IGA 的贡献。

## 2. 固定条件

所有运行的轨迹侧必须相同，并全部保持 A2G-aware：

```text
data_level = low
variant = A
algorithm = td3_bc（论文显示 TD3）
teacher = mapguide
guide_prob = 0.5
guide_min_prob = 0.2
guided_warmup_episodes = 400
bc_weight = 1.0
bc_min_weight = 0.25
bc_decay_episodes = 600
demo_fraction = 0.25
episodes = 3000
seed = 1
eval_interval = 25
outage_max_s = 6
objective_w1 = 0.5
objective_w2 = 0.5
device = cuda
```

禁止添加 `--a2g_blind`。本任务固定 A2G 可见性，只研究顺序。

## 3. 唯一顺序与复用规则

| 方法 | seed 1 顺序 | 顺序信息集 | 处理 |
|---|---|---|---|
| Proposed IGA | `[1,6,2,3,5,4]` | payload + G2A + A2G | 直接复用现有 6 s 主结果 |
| Distance-TSP | `[1,6,5,2,3,4]` | 纯几何 | 新跑 |
| EEGA | `[1,6,5,3,2,4]` | payload + G2A 互易吞吐 | 新跑 |
| Yang-style | `[1,6,5,4,2,3]` | payload + G2A 互易吞吐 | 新跑 |
| SA | `[1,6,5,2,3,4]` | G2A flight + outage | 与 Distance-TSP 顺序相同，不重复训练 |

SA 在 seed 1 与 Distance-TSP 得到同一顺序。固定训练 seed、轨迹网络和其余超参数
时，两者应产生完全相同的结果，因此 SA 只需在论文中说明顺序重合，不占用一次正式训练。

现有 IGA 直接复用：

```text
journal_results/v4-qos/w50-50/qos6/A/low/td3_bc
```

该目录已经满足：

```text
state_schema = v4-max-contiguous
qos_constraint = max_contiguous_outage
T+tau = 244.5 s
T_out^cont = 3.0 s
feasible evaluations = 108/120
```

## 4. 新运行矩阵

正式新增三组：

```text
journal_results/v4-sequence-ablation/a2g-aware/distance-tsp
journal_results/v4-sequence-ablation/a2g-aware/eega
journal_results/v4-sequence-ablation/a2g-aware/yang
```

输出目录中必须记录：

- 顺序算法；
- 实际巡检顺序；
- 顺序算法使用的信息集；
- 固定 A2G-aware 轨迹协议；
- 训练 seed；
- 地图哈希；
- 所有评价与最优结果。

## 5. 公共命令

Distance-TSP：

```bash
python Train_SINR.py \
  --variant A \
  --data_level low \
  --algorithm td3_bc \
  --teacher mapguide \
  --guide_prob 0.5 \
  --guide_min_prob 0.2 \
  --guided_warmup_episodes 400 \
  --bc_weight 1.0 \
  --bc_min_weight 0.25 \
  --bc_decay_episodes 600 \
  --demo_fraction 0.25 \
  --episodes 3000 \
  --seed 1 \
  --eval_interval 25 \
  --outage_max_s 6 \
  --objective_w1 0.5 \
  --objective_w2 0.5 \
  --sequence_method distance_tsp \
  --output_root journal_results/v4-sequence-ablation/a2g-aware/distance-tsp \
  --device cuda
```

EEGA：保持以上所有参数不变，仅替换：

```text
--sequence_method eega
--output_root journal_results/v4-sequence-ablation/a2g-aware/eega
```

Yang：保持以上所有参数不变，仅替换：

```text
--sequence_method yang
--output_root journal_results/v4-sequence-ablation/a2g-aware/yang
```

Win 应使用自己的 Python 3.11+ PyTorch 环境，不能用没有 Torch 的默认 Python。

## 6. 每组验收门槛

开始训练前：

- `state_schema` 以 `v4-max-contiguous` 开头；
- `qos_constraint = max_contiguous_outage`；
- `a2g_blind = false`；
- low payload 为 `[90,90,60,60,90,60]`；
- 地图哈希与其他 v4 运行一致。

训练结束后：

- 3000 episodes，seed 1，120 次周期评估；
- 顺序与该方法的 seed 1 顺序一致；
- 使用纯 Actor、无 MapGuide、无探索噪声的 `eval_best`；
- 只有 `eval_best.success=true` 且 `outage_time_s <= 6.0` 才可进入可行排名；
- 失败结果保留，不能换 seed、延长回合或修改预算后冒充原配置；
- 报告 `T`、`tau`、`T+tau`、`T_out^cont`、可行评估数和最优 episode。

## 7. 每组回传文件

```text
run_config.json
eval_history.json
UAV_TD3_improved.npz
preflight/
training_results.jpg
```

如需 Mac 复核轨迹，另把：

```text
models/*td3_actoreval_best.pth
```

复制到不受 `.gitignore` 影响的同步目录后推送，不要直接提交 `models/`。

## 8. 结果解读规则

主比较表应固定为：

| 轨迹栈 | A2G 可见 | 顺序 |
|---|---|---|
| TD3 | 是 | IGA |
| TD3 | 是 | Distance-TSP |
| TD3 | 是 | EEGA |
| TD3 | 是 | Yang-style |

只有当上述比较显示 IGA 的目标值更低，或同时改善 `T`/`tau` 且可靠性不差时，
才能写：

> IGA 的顺序优化在相同轨迹学习器下带来独立增益。

如果差距很小、排序不稳定，或 TSP 反而更低，应诚实写：

> A2G-aware trajectory learning is the dominant factor; the IGA sequence provides a secondary refinement.

不能把现有 A2G-blind 层面 3 结果当作顺序模块的单独证据。

## 9. 多 seed 条件

第一阶段只跑 seed 1 的三个新顺序。若前三名的差距接近或排序依赖单个 seed，
再补 seed 2、3：

```text
4 个唯一顺序 × 2 个额外 seed = 8 次新运行
```

单 seed 阶段不能写均值、标准差、置信区间或统计显著性。

## 10. 完成定义

以下全部满足才算顺序消融完成：

- Distance-TSP、EEGA、Yang 三组均有完整配置和评价历史；
- 三组均未使用 `--a2g_blind`；
- IGA 复用运行与本任务的其他条件完全一致；
- 至少三组新运行能给出有效 `eval_best`，失败项单独保留；
- 形成四方法同轨迹栈对照表；
- 明确回答 IGA 顺序是否带来独立增益；
- 结果和模型 actor 回传并推送。
