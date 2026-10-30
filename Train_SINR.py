import torch
from TD3 import TD3
from DDPG import DDPG
import ReplayBuffer
from World import World
from reward_logging import log_reward_terms
from training_config import ALGORITHMS, build_output_paths, load_fixed_payloads, resolve_algorithm
from sequence_runner import solve_ga_order
from training_transitions import replay_terminal_mask
from exploration import exploration_scales
from map_guidance import MapGuide
from demonstration import linear_decay
from mission_preflight import run_preflight
from evaluate_policy import STATE_SCHEMA, load_run_context, _build_world, rollout, selection_key
import json
import random
import numpy as np
import time
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
import argparse

def create_parser():
    """
    Parses each of the arguments from the command line
    :return ArgumentParser representing the command line arguments that were supplied to the command line:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav_h", help="set the UAV height", type=float,default=1.0)
    parser.add_argument("--gamma", help="set the gamma", type=float,default=0.995)
    parser.add_argument("--buffer", help="set the buffer", type=int,default=200000)
    parser.add_argument("--net_width", help="set the net width",type=int,default=256)
    parser.add_argument("--exploration_strategy", help="exploration strategy", type=str, default="guided")
    parser.add_argument("--min_exploration", help="minimum exploration noise", type=float, default=0.2)
    parser.add_argument("--max_exploration", help="maximum exploration noise", type=float, default=0.6)
    parser.add_argument("--data_level", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--episodes", help="training episode count", type=int, default=2000)
    parser.add_argument("--output_root", default="journal_results")
    parser.add_argument("--ga_iterations", type=int, default=500)
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="td3",
                        help="td3/ddpg are standard agents; td3_bc/ddpg_bc add MapGuide demonstrations and imitation")
    parser.add_argument("--outage_max_s", type=float, default=100.0)
    parser.add_argument("--train_memory_size", type=int, default=2000)
    parser.add_argument("--guided_warmup_episodes", type=int, default=400,
                        help="episodes over which the whole-episode MapGuide probability decays to zero")
    parser.add_argument("--guide_prob", type=float, default=None,
                        help="initial whole-episode MapGuide probability (*_bc only; default 0.5)")
    parser.add_argument("--bc_weight", type=float, default=None,
                        help="initial imitation weight on demonstrated samples (*_bc only; default 1.0)")
    parser.add_argument("--bc_min_weight", type=float, default=None,
                        help="persistent imitation floor (*_bc only; default 0.25)")
    parser.add_argument("--demo_fraction", type=float, default=None,
                        help="demonstration share of each replay batch (*_bc only; default 0.25)")
    parser.add_argument("--bc_decay_episodes", type=int, default=600,
                        help="episodes over which bc_weight decays to bc_min_weight")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--variant", choices=World.VARIANTS, default="baseline",
                        help="ablation: baseline (absolute heading), A (target-relative heading), AB (A + map neighborhood)")
    return parser

class AdaptiveExploration:
    """自适应探索策略"""
    def __init__(self, initial_noise=0.3, min_noise=0.05, decay_rate=0.9995, 
                 stagnation_threshold=50, boost_factor=1.5):
        self.initial_noise = initial_noise
        self.min_noise = min_noise
        self.current_noise = initial_noise
        self.decay_rate = decay_rate
        self.stagnation_threshold = stagnation_threshold
        self.boost_factor = boost_factor
        
        # 用于检测性能停滞
        self.reward_history = []
        self.stagnation_count = 0
        self.last_avg_reward = -float('inf')
        
    def update(self, episode_reward):
        self.reward_history.append(episode_reward)
        
        # 每50个episode检查一次是否停滞
        if len(self.reward_history) >= self.stagnation_threshold:
            current_avg = np.mean(self.reward_history[-self.stagnation_threshold:])
            
            # 如果性能提升不明显，增加探索
            if current_avg <= self.last_avg_reward + 1e-3:  # 阈值可调
                self.stagnation_count += 1
                if self.stagnation_count >= 3:  # 连续3次检测到停滞
                    self.current_noise = min(self.initial_noise, 
                                           self.current_noise * self.boost_factor)
                    self.stagnation_count = 0
                    print(f"检测到性能停滞，增加探索噪声至: {self.current_noise:.4f}")
            else:
                self.stagnation_count = 0
                
            self.last_avg_reward = current_avg
        
        # 正常衰减
        self.current_noise = max(self.min_noise, self.current_noise * self.decay_rate)
        
    def get_noise(self):
        return self.current_noise

class OrnsteinUhlenbeckNoise:
    """OU噪声，提供更平滑的探索"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2, dt=1e-2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.ones(size) * self.mu
        
    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        self.state += dx
        return self.state
    
    def reset(self):
        self.state = np.ones(self.size) * self.mu

def epsilon_greedy_exploration(model, s, epsilon, max_action):
    """ε-贪婪探索策略"""
    if np.random.random() < epsilon:
        # 随机动作
        action = np.random.uniform(-max_action, max_action)
        action[0] = np.random.uniform(-np.pi, np.pi)  # 角度
        action[1] = np.random.uniform(0, max_action[1])  # 距离
        return action
    else:
        return model.select_action(s)

def distance_based_exploration(current_pos, target_pos, base_noise, distance_factor=0.1):
    """基于距离的探索噪声调整"""
    distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
    # 距离目标越远，探索越大
    distance_multiplier = 1.0 + distance * distance_factor
    return min(base_noise * distance_multiplier, base_noise * 3.0)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def draw_location(x_uav, y_uav, t, savepath):
    x_uav = np.transpose(x_uav)
    y_uav = np.transpose(y_uav)
    plt.figure(facecolor='w', figsize=(20, 20))
    
    for i in range(uav_num):
        plt.plot(x_uav[i][0:t+1], y_uav[i][0:t+1], c='blue', marker='.', linewidth=3.5, markersize=7.5)
        plt.plot(x_uav[i][t], y_uav[i][t], c='green', marker='.', markersize=12.5)
    
    for index in range(world.urban_world.Build_num):
        x1 = world.HeightMapMatrix[index][0]
        x2 = world.HeightMapMatrix[index][1]
        y1 = world.HeightMapMatrix[index][2]
        y2 = world.HeightMapMatrix[index][3]
        XList = [x1, x2, x2, x1, x1]
        YList = [y1, y1, y2, y2, y1]
        plt.plot(XList, YList, 'r-')
    
    plt.title('location', fontsize=30)
    plt.xlim((0, Length))
    plt.ylim((0, Width))
    plt.grid()
    plt.savefig(savepath)
    plt.close()

# 参数解析
parser = create_parser()
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
gamma = args.gamma
net_width = args.net_width
buffer = args.buffer
uav_h = args.uav_h
exploration_strategy = args.exploration_strategy
min_exploration = args.min_exploration
max_exploration = args.max_exploration
data_level = args.data_level
algorithm = args.algorithm
profile = resolve_algorithm(algorithm, args.guide_prob, args.bc_weight,
                            args.bc_min_weight, args.demo_fraction)
args.guide_prob, args.bc_weight = profile["guide_prob"], profile["bc_weight"]
args.bc_min_weight, args.demo_fraction = profile["bc_min_weight"], profile["demo_fraction"]
outage_max_s = args.outage_max_s
train_memory_size_arg = args.train_memory_size
guided_warmup_episodes = args.guided_warmup_episodes
payloads = load_fixed_payloads(data_level)
output_paths = build_output_paths(args.output_root, data_level, algorithm, variant=args.variant)

train_path = str(output_paths["logs"])
mkdir(train_path)
writer = SummaryWriter(train_path)

# 环境参数
uav_num = 1
user_num = 6
T = 1000
Length = 20
Width = 20
a_dim = uav_num*2

V_max = 0.50
delta_t = 0.5
dist_max = delta_t * V_max
distance_noise_cap = min(0.05, dist_max)

max_action = np.array([math.pi, dist_max])
min_action = np.array([-math.pi, 0])
total_episode = args.episodes
sample_episode = 1000

# 改进的训练参数
train_memory_size = min(train_memory_size_arg, buffer // 10)
train_freq = 2  # 每4步训练一次
data_size = 300

ini_loc = [2.62, 2.65]
end_loc = [18.95, 11.12]
BS_loc = np.array([[4.77,4.56,0.25],[14.89,4.89,0.25],[4.63,13.78,0.25],[13.48,14.89,0.25]])
trave_order = solve_ga_order(payloads, iterations=args.ga_iterations)

world = World(length=Length, width=Width, uav_num=uav_num, dist_max=dist_max, delta_t=delta_t, t=T, uav_h=uav_h,
              data_size=data_size, ini_loc=ini_loc, end_loc=end_loc,
              users_name='results/datas/Users_'+str(user_num)+'.txt',
              BS_loc=BS_loc, traverse_sequence=trave_order, data_sizes=payloads, outage_max_s=outage_max_s,
              variant=args.variant)
s_dim = world.state_dim

preflight = run_preflight(data_level, output_paths["root"] / "preflight", order=trave_order,
                          outage_max_s=outage_max_s, total_steps=T)
print("Mission preflight:", {k: preflight[k] for k in ("success", "time_s", "outage_s", "uploaded_mb", "visited_points")}, flush=True)
writer.flush()
if not preflight["success"]:
    writer.close()
    raise RuntimeError("Full mission preflight failed; training was not started. See preflight.json.")
if args.preflight_only:
    writer.close()
    raise SystemExit(0)

(output_paths["root"] / "run_config.json").write_text(json.dumps({
    **vars(args), "state_dim": s_dim, "order": trave_order, "payloads_mb": payloads.tolist(),
    "map_hashes": preflight["map_hashes"], "sequence_method": "GA with exact verification/fallback",
    "state_schema": STATE_SCHEMA,
    "model_selection": "eval_best: deterministic guide-free evaluation, feasible first, then minimum T + tau",
    "reward_schema": "v4-seconds: -(T + tau) + 2x progress + milestones; T_out only as hard constraint; x0.01",
    "base_agent": profile["base"],
    "demonstration": ("none" if profile["guide_prob"] == 0.0 else
                      "whole-episode MapGuide with decaying probability; persistent demo replay; behavior cloning weight decays to a nonzero floor"),
    "demo_buffer_capacity": int(buffer),
    "demo_sampling_fraction": profile["demo_fraction"],
    "boundary_handling": "per-coordinate clip to map bounds with boundary penalty",
}, indent=2), encoding="utf-8")
# A separate world keeps evaluation from disturbing the training episode state.
eval_context = load_run_context(output_paths["root"])
eval_world = _build_world(eval_context)
eval_best_report = None
eval_history = []

# Legacy terrain constructors seed NumPy internally; restore the run seed.
np.random.seed(args.seed)
random.seed(args.seed)

# 初始化探索策略
if exploration_strategy == "adaptive":
    explorer = AdaptiveExploration(initial_noise=max_exploration, min_noise=min_exploration)
elif exploration_strategy == "ou_noise":
    ou_noise = OrnsteinUhlenbeckNoise(a_dim, sigma=0.2)
    expl_noise = max_exploration
else:  # 线性衰减
    expl_noise = max_exploration

# TD3模型初始化
env_with_Dead = True
kwargs = {
    "env_with_Dead": env_with_Dead,
    "state_dim": s_dim,
    "action_dim": a_dim,
    "max_action": max_action,
    "train_path": train_path,
    "gamma": gamma,
    "net_width": net_width,
    "a_lr": 1e-4,
    "c_lr": 1e-3,
    "Q_batchsize": 256,
}
model = TD3(**kwargs) if profile["base"] == "td3" else DDPG(**kwargs)
model_path = str(output_paths["model_prefix"])
mkdir(str(output_paths["model_prefix"].parent))
replay_buffer = ReplayBuffer.ReplayBuffer(
    s_dim, a_dim, max_size=int(buffer), demo_max_size=int(buffer),
    demo_fraction=profile["demo_fraction"])

# 早停和最佳模型保存
best_reward = -float('inf')
patience = 200
no_improve_count = 0

# 记录变量
ep_rewards = []
ep_exploration_noise = []
x0_uav = np.zeros([total_episode+1, T + 1, world.uav_num])
y0_uav = np.zeros([total_episode+1, T + 1, world.uav_num])
z0_uav = np.zeros([total_episode+1, T + 1, world.uav_num])
location = np.zeros([3, world.uav_num])

print(f"开始训练，探索策略: {exploration_strategy}")
print(f"初始探索噪声: {max_exploration}, 最小探索噪声: {min_exploration}")

# 主训练循环
for episode in tqdm(range(1, total_episode+1), ascii=True, unit='episodes'):
    t3 = time.time()
    s, t = world.reset()
    map_guide = MapGuide(world)
    # Whole demonstration episodes avoid a guide silently repairing actor mistakes.
    guide_probability = linear_decay(args.guide_prob, episode - 1, guided_warmup_episodes) if exploration_strategy == "guided" else 0.0
    guided_episode = np.random.random() < guide_probability
    model.bc_weight = linear_decay(
        args.bc_weight, episode - 1, args.bc_decay_episodes,
        floor=args.bc_min_weight)
    fly_dis = 0.0
    data_total = 0.0
    
    
    # 重置OU噪声
    if exploration_strategy == "ou_noise":
        ou_noise.reset()
    
    # 记录初始位置
    for i, uav in enumerate(world.UAVs):
        location[0][i] = uav.x
        location[1][i] = uav.y
        location[2][i] = uav.h
    x0_uav[episode][0] = location[0]
    y0_uav[episode][0] = location[1]
    z0_uav[episode][0] = location[2]
    
    sum_reward = 0
    sum_fly_v = 0
    done = False
    step_count = 0
    angle_noise_sum = distance_noise_sum = 0.0
    guidance_steps = 0
    
    while not done:
        step_count += 1
        
        # 选择动作和探索策略
        target_distance = np.linalg.norm(np.array([world.UAVs[0].x, world.UAVs[0].y]) - world.target_loc)
        angle_noise, distance_noise = exploration_scales(episode, total_episode, target_distance)

        if exploration_strategy == "guided":
            if guided_episode:
                guidance_steps += 1
                # MapGuide plans absolute headings; store them in the policy's action frame.
                a = world.absolute_to_action(map_guide.action(angle_noise, distance_noise, np.random))
            else:
                a = model.select_action(s) + np.random.normal(0, [angle_noise, distance_noise], size=a_dim)
            current_noise = angle_noise
        elif exploration_strategy == "adaptive":
            current_noise = explorer.get_noise()
            current_noise = min(current_noise, angle_noise)
            a = model.select_action(s) + np.random.normal(0, [current_noise, distance_noise], size=a_dim)
            
        elif exploration_strategy == "ou_noise":
            noise = ou_noise.sample() * np.array([angle_noise, distance_noise])
            a = model.select_action(s) + noise
            current_noise = expl_noise
            
        elif exploration_strategy == "epsilon_greedy":
            epsilon = max(min_exploration, max_exploration * (1 - episode / total_episode))
            a = epsilon_greedy_exploration(model, s, epsilon, max_action)
            current_noise = epsilon
            
        else:  # 线性衰减
            current_noise = max(min_exploration, max_exploration * (1 - episode / total_episode))
            a = model.select_action(s) + np.random.normal(0, [angle_noise, distance_noise], size=a_dim)
        
        # 动作裁剪
        a = np.clip(a, min_action, max_action)
        angle_noise_sum += angle_noise
        distance_noise_sum += distance_noise
        
        fly_dis += a[1]
        sum_fly_v += a[1] * 100 / delta_t
        
        s_, r, done, t, terminal = world.step_inside(a, s, t)
        sum_reward += r
        data_total += world.last_uploaded_data
        replay_buffer.add(s, a, r, s_, replay_terminal_mask(done), demo=guided_episode)
        
        # 训练模型
        if replay_buffer.total_size > train_memory_size and step_count % train_freq == 0:
            model.train(replay_buffer)
        
        s = s_
        
        # 记录位置
        for i, uav in enumerate(world.UAVs):
            location[0][i] = uav.x
            location[1][i] = uav.y
            location[2][i] = uav.h
        x0_uav[episode][t] = location[0]
        y0_uav[episode][t] = location[1]
        z0_uav[episode][t] = location[2]
    
    print("Reward_per_Episode:",sum_reward,"Data_Volume:",data_total)
    
    
    # 更新探索策略
    if exploration_strategy == "adaptive":
        explorer.update(sum_reward)
        current_noise = explorer.get_noise()
    elif exploration_strategy == "ou_noise":
        expl_noise = max(min_exploration, expl_noise * 0.9995)
        current_noise = expl_noise
    
    # 记录和保存
    ep_rewards.append(sum_reward)
    ep_exploration_noise.append(current_noise)
    if episode %50 == 0 and episode >0:
        model.save(episode,model_path)
    
    # 早停检查
    if sum_reward > best_reward:
        best_reward = sum_reward
        model.save("best", model_path)
        no_improve_count = 0
    else:
        no_improve_count += 1
    
    # 记录到tensorboard
    writer.add_scalar('total_reward', sum_reward, episode)
    writer.add_scalar('exploration_noise', current_noise, episode)
    writer.add_scalar('average_speed', sum_fly_v/max(t, 1), episode)
    writer.add_scalar('path_efficiency', 
                     np.linalg.norm(np.array(end_loc) - np.array(ini_loc)) / max(fly_dis, 1e-6), 
                     episode)
    log_reward_terms(writer, world.reward_summary(), episode)
    if episode % args.eval_interval == 0 or episode == total_episode:
        report, _ = rollout(eval_context, eval_world, model.select_action)
        eval_history.append({"episode": episode, **{k: report[k] for k in (
            "success", "termination_reason", "visited_inspection_points", "uploaded_mb",
            "task_time_s", "transmission_delay_s", "outage_time_s", "total_return",
            "boundary_violations", "stall_steps", "max_stall_run")}})
        writer.add_scalar('eval/success', float(report["success"]), episode)
        writer.add_scalar('eval/visited_inspection_points', report["visited_inspection_points"], episode)
        writer.add_scalar('eval/objective_s', report["task_time_s"] + report["transmission_delay_s"], episode)
        writer.add_scalar('eval/outage_time', report["outage_time_s"], episode)
        writer.add_scalar('eval/boundary_collisions', report["boundary_violations"], episode)
        writer.add_scalar('eval/max_stall_run', report["max_stall_run"], episode)
        if eval_best_report is None or selection_key(report) < selection_key(eval_best_report):
            eval_best_report = {**report, "episode": episode}
            model.save("eval_best", model_path)
        (output_paths["root"] / "eval_history.json").write_text(json.dumps(
            {"eval_best": {k: v for k, v in eval_best_report.items() if k not in ("visits", "reward_terms", "map_hashes")},
             "history": eval_history}, indent=2, ensure_ascii=False), encoding="utf-8")
    writer.add_scalar('episode/min_target_distance', world.min_target_distance, episode)
    writer.add_scalar('episode/outage_time', world.out_time, episode)
    writer.add_scalar('episode/visited_inspection_points', world.Traverse_order, episode)
    writer.add_scalar('episode/boundary_collisions', world.fa, episode)
    writer.add_scalar('episode/stall_steps', world.stall_steps, episode)
    writer.add_scalar('episode/max_stall_run', world.max_stall_run, episode)
    writer.add_scalar('exploration/angle_std_mean', angle_noise_sum / step_count, episode)
    writer.add_scalar('exploration/distance_std_mean', distance_noise_sum / step_count, episode)
    writer.add_scalar('exploration/guidance_fraction', guidance_steps / step_count, episode)
    writer.add_scalar('exploration/guide_probability', guide_probability, episode)
    writer.add_scalar('training/bc_weight', model.bc_weight, episode)
    writer.add_scalar('training/online_buffer_size', replay_buffer.size, episode)
    writer.add_scalar('training/demo_buffer_size', replay_buffer.demo_size, episode)
    writer.add_scalar('training/demo_fraction', replay_buffer.demo_fraction, episode)
    writer.add_scalar('training/critic_updates', model.q_iteration, episode)
    writer.add_scalar('training/actor_updates', model.a_iteration, episode)
    
    if episode % 50 == 0:
        model.save(episode, model_path)
        print(f'Episode: {episode}, Epoch: {t}, Reward: {sum_reward:.4f}, '
              f'Noise: {current_noise:.4f}, Best: {best_reward:.4f}')
    
    # 早停
    # if no_improve_count > patience:
    #     print(f"早停于episode {episode}, 最佳奖励: {best_reward:.4f}")
    #     break

# 绘制结果
def get_moving_average(mylist, N):
    if len(mylist) < N:
        return mylist
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            moving_aves.append(moving_ave)
    return moving_aves

# 奖励曲线
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(range(len(ep_rewards)), ep_rewards, alpha=0.6, label='Episode Reward')
N = min(200, len(ep_rewards) // 4)
if N > 0:
    return_mov_avg = get_moving_average(ep_rewards, N)
    ax1.plot(np.arange(len(return_mov_avg)) + N, return_mov_avg, 'r-', 
             linewidth=3, label=f'Moving Average ({N})')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Training Reward Curve')
ax1.legend()
ax1.grid()

# 探索噪声曲线
ax2.plot(range(len(ep_exploration_noise)), ep_exploration_noise, 'g-', linewidth=2)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Exploration Noise')
ax2.set_title('Exploration Noise Decay')
ax2.grid()

plt.tight_layout()
plt.savefig(output_paths["root"] / 'training_results.jpg', dpi=300, bbox_inches='tight')
plt.close()

# 保存数据
if len(ep_rewards) > 200:
    return_mov_avg = get_moving_average(ep_rewards, 200)
    np.savez(output_paths["root"] / 'UAV_TD3_improved', return_mov_avg=return_mov_avg, 
             ep_rewards=ep_rewards, exploration_noise=ep_exploration_noise)
else:
    np.savez(output_paths["root"] / 'UAV_TD3_improved', ep_rewards=ep_rewards, 
             exploration_noise=ep_exploration_noise)

writer.close()
print("训练数据已保存到 UAV_TD3_improved.npz")
print(f"最佳模型奖励: {best_reward:.4f}")
