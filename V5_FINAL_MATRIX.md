# V5 Final Experiment Matrix

更新：2026-11-08

## 复用规则

- 旧 PRIGA 边权已恢复；
- E1 CA-TD3 三档全部复用；
- E2 最终保留 PRIGA、Yang-style、EEGA、DRL-SA、SA-D3QN 和 EEAP-PSO；
- E2 Medium/High 不重新训练轨迹算法，只计算顺序并用现有 CA-TD3 actor 重放；
  停滞时由 MapGuide 接管；若混合控制在任务完成前终止，则明确执行整段
  MapGuide restart fallback；
- E3 Low 复用，只新增 Medium/High；
- E4 Low 复用，只新增 Medium/High；
- E5 DRM-HO 三档和六个基线 Low 全部复用；
- E5 MCO-A* Medium/High 已在 Mac 完成；
- E5 其余五个学习型基线只新增 Medium/High。

## 实验矩阵

| ID | 实验内容 | 已完成/复用 | Mac 新增 | Win 新增 |
|---|---|---|---|---|
| E1 | 固定 PRIGA 的轨迹执行器比较 | CA-TD3 L/M/H；D3QN Low；PPO/SAC Low | — | PPO M/H；D3QN M/H；SAC M/H（DDPG 已从最终稿移除） |
| E2 | CA-TD3 + MapGuide assisted completion，比较巡检顺序算法 | Low 全部复用；PRIGA M/H 复用 | GA/PSO/SA/A*/EEGA/Yang 的 M/H 顺序计算与 assisted replay | — |
| E3 | 固定 Distance-TSP 的 A2G 消融 | Low blind/visible | Medium/High blind/visible | — |
| E4 | QoS 与目标权重稀疏扫描 | Low 全部；M/H balanced 6 s | M/H 其余 12 个稀疏条件 | — |
| E5 | 文献基线比较 | DRM-HO 三档；六基线 Low；MCO M/H | 五个学习型基线 M/H | — |

## 增量训练数量

### Win

```text
PPO Low/Medium/High          3
D3QN Medium/High             2
DDPG Low/Medium/High         3
SAC Low/Medium/High          3
--------------------------------
合计                         11
```

### Mac

```text
E3 Medium/High               4
E4 Medium/High sparse        12
E5 five baselines Medium/High 10
--------------------------------
合计                          26
```

E2 不包含神经网络重新训练，只有 12 个 Medium/High assisted replay；
与 Low 的 6 个结果共同构成最终 18 行顺序比较。
E2 结果必须同时报告接管时刻、MapGuide 控制步数以及是否使用了整段重启兜底。

## V5 新训练统一约束

- `N=2000` slots，即 1000 s；
- `qos_constraint=max_contiguous_outage`；
- `T_out^max=6 s`；
- `kappa_out=0.5`；
- `qos_enforcement=continue`；
- 完成任务后单独记录 `Complete` 和 `Constraint`；
- 正式 actor 评估无 MapGuide、无探索噪声；
- `eval_best` 选择顺序为：可行完整、违约完整、最大进度；
- seed 1。

## Win 启动

```bash
python run_v5_matrix.py \
  --groups E1 \
  --levels low medium high \
  --device cuda \
  --workers 3 \
  --reuse-existing-e1 \
  --execute
```

## 当前状态

E1-E5 的最终论文数据均已归档，无待运行训练。E1 Medium/High actor 与轨迹
已由 Win commit `7529dbc` 同步；最终论文使用同一批 validated rollout，
SAC Medium 采用 actor 归档轨迹指标 `tau=328.0 s`、`T_out^cont=20.5 s`。

E4 新增 Medium/High 均使用 V5 `qos_enforcement=continue` 和 1000 s
（2000-slot）horizon。表中 4/6 的未完成结果都耗尽 horizon；QoS 违约只记录，
不会提前终止 rollout。因此这些结果不是由 v4 严格约束提前截断造成，无需按
原配置盲目重跑。Medium/QoS=4 s 是约束满足但 4/6 超时的直接例证。

最终论文的空间轨迹图只展示 Low：五档 QoS 轨迹叠加 G2A coverage map，
三组目标权重轨迹叠加 A2G SNR map。Medium/High 的完整数值结果仍保留在
QoS 表与权重表中，但不绘制空间轨迹，以避免用独立训练造成的经验倒挂解释
数据量难度。

