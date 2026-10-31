# 项目交接文档（2026-10-31）

## 1. 项目是什么

在 ICC 2026 会议论文（双无线电地图 UAV 巡检）基础上做的期刊扩展。核心变量是**六个巡检点数据量不同**，分低/中/高三档固定负载。要优化两件事：

- **巡检顺序**：数据感知 GA（实际是"GA + 穷举校验回退"）。
- **飞行轨迹**：TD3（论文主方法），对比 DDPG、D3QN、A*、MapGuide。

目标：最小化 `T + τ`（完成时间 + 上传时延）。
硬约束：G2A 中断时间 `T_out ≤ 100 s`（超过即判失败）。

## 2. 环境

- **Python**：`D:/Anaconda/envs/pytorch/python.exe`（Python 3.11、PyTorch 2.6 cu124、CUDA 可用）。
  - 默认 `python` 是 3.14，没有 torch 和 pytest，**不要用它**。
- **跑测试**：`D:/Anaconda/envs/pytorch/python.exe -m pytest -q`（当前 152 个测试全过）。
- **Git 远程**：
  - `origin` → `https://github.com/LanceGan/SingleUAV.git`（旧仓库，未动）。
  - `new` → `https://github.com/LanceGan/Single_UAV_new.git`（本次推送目标）。
  - 本机配置了 http/https 代理 `127.0.0.1:7897`；代理关闭时 push 会失败。

## 3. 关键结果（截至交接时）

### TD3-BC（formal-v5-restart，3000 回合，MapGuide 示范，引导衰减到 0）

每 25 回合纯 actor 评估 120 次，报告 `eval_best`（无 MapGuide、无噪声）：

| 档位 | 成功次数 | T (s) | τ (s) | T+τ (s) | T_out (s) |
|---|---|---|---|---|---|
| 低 | 64/120 | 165.5 | 78.5 | **244.0** | 18.5 |
| 中 | 96/120 | 222.5 | 131.0 | **353.5** | 20.0 |
| 高 | 39/120 | 300.5 | 200.5 | **501.0** | 17.5 |

- DDPG-BC 低负载：**0 次成功**（Q 值高估，方法本身失败，作为对比结论）。
- 低/中负载后期稳定；低负载第 2075 回合后、高负载第 2625 回合后出现过退化，`eval_best` 不受影响。

### 基线

| 方法 | 低 T+τ | 中 T+τ | 高 T+τ | 说明 |
|---|---|---|---|---|
| **A\***（逐段最优，World 重放） | **214.0** | **318.0** | **447.0** | T_out ≈ 24.5–27.5 |
| MapGuide 规划器 | 273.0 | 369.0 | 492.0 | Dijkstra + 上传点启发式 |

A* 比 TD3 好 11–14%。原因是 A* 直接知道整张地图并精确优化 T+τ，但高负载一次要搜索 463 s；TD3 决策只需毫秒级。

## 4. 刚做的改动（已提交，未训练验证）

### 4.1 A* 作为所有 DRL 的示范（commit `e18d8d3`）

之前示范用的是 MapGuide，现在默认改成 A* 的最优解：

- **TD3-BC / DDPG-BC**：新增 `--teacher {astar,mapguide}`，默认 `astar`。示范回合直接重放 A* 路径（动作已转换到策略坐标系）。
- **D3QN**：新增 A* 示范。重放 A* 最优路径，每步把连续动作**快照成最近的离散动作**，用 DQfD 大边界损失（`--demo_weight 1.0`、`--demo_margin 0.8`）让 argmax 朝示范靠拢。
- 示范源在 `astar_teacher.py`，规划一次后缓存到 `journal_results/astar-teacher/teacher_{low,medium,high}.json`（已预生成三档，训练启动不再等待搜索）。

### 4.2 示范全程保留（commit `3b4b40d`）

TD3-BC/DDPG-BC 的整回合示范概率在前 400 回合从 0.5 线性降到 **0.2，之后保持 0.2**（不是降到 0）。行为克隆权重在前 600 回合从 1.0 降到 0.25 后保持。示范数据单独存放，不被新数据挤掉。

### 4.3 之前已修好的关键项

- 地图轴顺序（`[y, x]`）、GA 点位（`Users_6.txt`）、GA 航段单位。
- 越界改为逐坐标裁剪（沿墙滑动），保留罚分；有停滞/碰撞诊断。
- 奖励按秒对齐 `T+τ`：`-dt`（T）、`-dt·1[待传]`（τ）、推进项、里程碑；T_out 只作硬约束。
- 状态/动作三种消融：`baseline`（绝对航向）、`A`（相对目标方位航向）、`AB`（A + 8 方向地图采样）。当前都用 `A`。
- TD3 目标值在 `no_grad` 下计算；QoS 与到达同一步时判失败；reset 清除成功标志。

## 5. 各方法怎么跑

### TD3-BC / DDPG-BC（A* 示范）

```bash
PY=D:/Anaconda/envs/pytorch/python.exe
$PY Train_SINR.py --variant A --data_level low  --algorithm td3_bc  --teacher astar --seed 1 --episodes 3000 --output_root journal_results/formal-v7
$PY Train_SINR.py --variant A --data_level medium --algorithm td3_bc --teacher astar --seed 1 --episodes 3000 --output_root journal_results/formal-v7
$PY Train_SINR.py --variant A --data_level high  --algorithm td3_bc  --teacher astar --seed 1 --episodes 3000 --output_root journal_results/formal-v7
$PY Train_SINR.py --variant A --data_level low  --algorithm ddpg_bc --teacher astar --seed 1 --episodes 3000 --output_root journal_results/formal-v7
```

标准版（无示范）用 `--algorithm td3` / `--algorithm ddpg`（不能开示范参数）。

### D3QN（A* 示范，DQfD 边界损失）

```bash
$PY Train_D3QN.py --data_level low --teacher astar --seed 1 --episodes 3000 --output_root journal_results/d3qn-baseline
# medium / high 同理改 --data_level
```

### A* 基线

```bash
$PY run_astar.py --mode sequential --output-dir journal_results/astar-baseline
```

### 评估已有结果

```bash
$PY evaluate_policy.py --run_dir journal_results/formal-v5-restart/A/low/td3_bc   # 连续算法
$PY evaluate_d3qn.py --run_dir journal_results/d3qn-baseline/low/d3qn             # D3QN
```

## 6. 结果目录约定

- 训练输出：`journal_results/<run>/A/<level>/<algorithm>/`，含 `run_config.json`、`eval_history.json`、`models/`（`eval_best`）、`evaluation/`。
- 论文图：`plot_formal_results.py` 生成；摘要见 `journal_results/formal-v5-restart/paper_figure_summaries.json`。
- 判断方法好坏一律看**纯 actor 的 `eval_best`**，不看训练奖励曲线（曲线混了示范回合；TensorBoard 里有 `total_reward_by_source/actor` 与 `/mapguide` 分开）。

## 7. 待办 / 下一步

1. **重训 TD3-BC × 3 档 + DDPG-BC 低负载，用 A\* 示范**（命令见 §5，输出建议 `formal-v7`）。A* 是更强的老师，TD3 有望接近或超过 A* 的 214/318/447。
2. **重训 D3QN × 3 档，用 A\* 示范**。之前 D3QN 无示范、未正式训练。
3. **高负载**是难点：TD3（MapGuide 老师）曾比 MapGuide 还差 1.8%，A* 老师下需重看。
4. **论文结论建议**：TD3 vs DDPG（DDPG 因 Q 高估失败）；TD3 vs A*（DRL 在线决策快、A* 离线搜索慢但更优）；TD3 vs MapGuide；三档负载下 T、τ、T_out 随数据量变化；GA 顺序随负载变化。
5. 若需消融：`--teacher mapguide`、`--variant baseline/AB`、`--algorithm td3`（无示范标准版）。

## 8. 文件地图

| 文件 | 作用 |
|---|---|
| `World.py` | 环境：地图查询、上传/QoS 记账、奖励、越界、状态/动作（variant） |
| `Train_SINR.py` | TD3/DDPG 训练入口（示范、评估、eval_best） |
| `TD3.py` / `DDPG.py` | 连续动作算法（含行为克隆项） |
| `D3QN.py` / `Train_D3QN.py` / `discrete_actions.py` / `evaluate_d3qn.py` | 离散动作 D3QN（含 DQfD 示范） |
| `astar_planner.py` / `run_astar.py` | QoS 约束 A\*（25 m 栅格，逐段/整体模式） |
| `astar_teacher.py` | A\* 示范缓存与重放、D3QN 示范快照 |
| `map_guidance.py` / `mission_preflight.py` | MapGuide 规划器、训练前预检 |
| `sequence_runner.py` / `GA_EQTSP.py` | 数据感知 GA 顺序 |
| `evaluate_policy.py` | 连续策略无噪声评估、轨迹图 |
| `training_config.py` / `training_transitions.py` / `reward_logging.py` / `exploration.py` | 配置/回放/奖励日志/探索 |
| `plot_formal_results.py` | 论文图表 |

## 9. 已知注意点

- `journal_results/` 已被 `.gitignore` 忽略，结果不提交；教师缓存也在里面。
- 工作区有杂项未跟踪文件：`.vscode/`、`code.zip`、`tmp_matplotlib_*.png`、一份论文 PDF；按需处理，勿混进提交。
- 论文若写"TD3"=带 A\* 示范的 TD3-BC，方法部分需说明训练使用了离线 A\* 解做示范（用户已确认这样写）。
- 所有 DRL 现在都带示范，才能公平比较 TD3 / DDPG / D3QN。
