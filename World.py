from entity import UAV, User
import numpy as np
import os
import sys
import math
import copy
from fixed_radio_maps import FixedRadioMaps
from numpy import linalg as LA
from rural_world import Rural_world


class World(object):
    # Reward terms are designed in seconds of the objective T + tau, then
    # scaled once by REWARD_SCALE so returns stay in a critic-friendly range.
    REWARD_SCALE = 0.01
    UPLOAD_LEG_REWARD = 10.0   # earned in full when one leg's payload is cleared
    ARRIVAL_REWARD = 50.0      # reaching a waypoint with its upload complete
    COMPLETION_REWARD = 200.0
    INFEASIBLE_PENALTY = 1000.0  # QoS violation, timeout, or early arrival
    BOUNDARY_SCALE = 0.01
    PROGRESS_WEIGHT = 2.0  # full-speed approach nets +dt per slot, hovering -dt
    # Ablation variants: baseline = absolute heading; A = heading relative to the
    # target bearing; AB = A plus dual-map samples on a target-frame ring.
    VARIANTS = ("baseline", "A", "AB")
    BASE_STATE_DIM = 8
    NEIGHBOR_DIRECTIONS = 8
    STALL_DISTANCE = 1e-3  # slots moving less than this count as motionless
    def __init__(self, length=10, width=10, uav_num=1, user_num=10,dist_max=0.10,
                 delta_t=0.5,t=200, uav_h=1,
                 data_size = 250, data_sizes=None, outage_max_s=100.0, ini_loc=[10.93,4.4],end_loc=[15.5,18.8],
                 users_name='Users.txt'
                 ,BS_loc=[],
                 traverse_sequence=[], variant="baseline"):
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {self.VARIANTS}")
        self.variant = variant
        self.relative_heading = variant in ("A", "AB")
        self.map_neighborhood = variant == "AB"
        self.state_dim = self.BASE_STATE_DIM + (2 * self.NEIGHBOR_DIRECTIONS if self.map_neighborhood else 0)
        self.length = length # 区域大小
        self.width = width
        self.uav_num = uav_num  # 总的无人机数目
        self.users_path = users_name # 用户位置存储路径
        self.user_num = user_num  # 用户数目
        self.Users = []
        self.UAVs = []
        self.T = t # 时间步
        self.t = 0
        self.max_x = length # 区域边界
        self.min_x = 0
        self.max_y = width
        self.min_y = 0
        self.uav_h = uav_h  # 无人机飞行高度
        self.dist_max = dist_max # max flying distance m
        self.delta_T = delta_t # 时隙长度
        self.fa = 0.0  # 出界
        self.r = 0.0  # 奖励
        self.reward_term_totals = self._empty_reward_terms()
        self.terminal = False
        self.total_engy = 0.0
        self.out_time = 0.0
        self.outage_max_s = outage_max_s
        self.NON_COMM_PENALTY = -5 # 不满足通信阈值的惩罚
        self.NON_COVER_PENALTY = -10 # 不满足覆盖阈值的惩罚
        
        self.Engy_w = 0.05

        self.initial_loc = ini_loc # 起点
        self.end_loc = end_loc # 终点
        self.distance = 0.5 # m
        self.SIR_THRESHOLD_COMM = 3 # dB 通信阈值
        self.SIR_THRESHOLD_COVER = 2 # dB 通信阈值
        self.data_size_ini = data_size # 每个巡检点的数据量
        self.data_size = self.data_size_ini  # 每个巡检点的数据量
        self.data_sizes = data_sizes
        self.current_payload = 0.0
        self.data_rate = 0
        self.last_uploaded_data = 0.0
        self.BandWidth = 1
        self.radio_maps = FixedRadioMaps('results/datas')
        self.a2g_snr_min = float(self.radio_maps.a2g_snr.min())
        self.a2g_snr_max = float(self.radio_maps.a2g_snr.max())
        self.current_a2g_snr = 0.0
        self.current_g2a_outage_probability = 0.0
        self.transmit = False # 是否完成数据传输
        self.trans_delay_start = 0
        self.trans_delay_end = self.T
        self.trans_delay = 0
        
        self.Traverse = traverse_sequence # 巡检点序列
        self.Traverse_order = 0 # 巡检点序列索引
        
        # =============Define the GT Distribution=======================
        
        self.GT_loc = BS_loc # 用户位置
        self.set_users()
        if self.data_sizes is None:
            self.data_sizes = np.full(len(self.Users), float(self.data_size_ini))
        else:
            self.data_sizes = np.asarray(self.data_sizes, dtype=float)
            if len(self.data_sizes) != len(self.Users):
                raise ValueError("data_sizes must provide one payload for each inspection user")
        self.max_payload = max(float(np.max(self.data_sizes)), 1.0)
        self.target = self.Users[self.Traverse[self.Traverse_order]-1]
        self.target_loc = np.array([self.target.x, self.target.y])  # 目标位置
        self.target_dis = LA.norm(self.initial_loc - self.target_loc) # 距离目标的距离
        self.min_target_distance = self.target_dis
        self.target_pre_loc = np.array([self.initial_loc[0],self.initial_loc[1]]) #上一个目标的位置，初始为起点
        
        
        self.urban_world = Rural_world(self.GT_loc)
        self.HeightMapMatrix = self.urban_world.Buliding_construct()
        

    def set_users(self):
        self.Users =[]
        if os.path.exists(self.users_path): # 读写文件 # 读入用户位置
            f = open(self.users_path, 'r')
            if f:
                user_loc = f.readline()
                user_loc = user_loc.split(' ')
                self.Users.append(User(float(user_loc[0]), float(user_loc[1]),float(user_loc[2])))
                # self.GT_loc[len(self.Users) - 1] = np.array(
                #     [float(user_loc[0]), float(user_loc[1]), float(user_loc[2])])
                while user_loc:
                    user_loc = f.readline()
                    if user_loc:
                        user_loc = user_loc.split(' ')
                        self.Users.append(User(float(user_loc[0]), float(user_loc[1]),float(user_loc[2])))
                        # self.GT_loc[len(self.Users) - 1] = np.array([float(user_loc[0]), float(user_loc[1]), float(user_loc[2])])
                f.close()
        else:
           assert False, "Users file not found: " + self.users_path

    def payload_for_user(self, user_index):
        return float(self.data_sizes[user_index - 1])

    def reset_target(self):
        self.Traverse_order = 0
        self.target = self.Users[self.Traverse[self.Traverse_order]-1]
        self.target_loc = np.array([self.target.x, self.target.y])  # 目标位置
        self.target_dis = LA.norm(self.initial_loc - self.target_loc) # 距离目标的距离
        self.min_target_distance = self.target_dis
        self.target_pre_loc = np.array([self.initial_loc[0],self.initial_loc[1]]) #上一个目标的位置，初始为起点
        
    # 重置位置，各种时间清0
    def reset(self):
        self.reset_target()
        self.set_uavs_loc()
        self.terminal = False
        self.stall_steps = 0
        self.max_stall_run = 0
        self._stall_run = 0
        self.trans_delay_start = 0
        self.trans_delay_end = self.T
        self.trans_delay = 0
        self.data_size = 0
        self.last_uploaded_data = 0.0
        self.transmit = True  
        self.t = 0
        self.fa = 0
        self.out_time = 0
        self.total_engy = 0.0
        self.reward_term_totals = self._empty_reward_terms()
        initial_location = np.array([[self.UAVs[0].x, self.UAVs[0].y]])
        self.current_a2g_snr = self.get_date_rate_A2G(initial_location)
        self.current_g2a_outage_probability = self.get_g2a_outage_probability(initial_location)
        state = self.reset_state()
        return state,self.t

    # def reset_loc(self):
    #     self.set_uavs_loc()
    #     state = self.reset_state()
    #     self.t = 0
    #     self.fa = 0
    #     self.out_time = 0
    #     self.total_engy = 0.0
    #     return state,self.t

    def set_uavs_loc(self):
        self.UAVs = []
        for i in range(self.uav_num):
            x = self.initial_loc[0]
            y = self.initial_loc[1]
            h = self.uav_h
            self.UAVs.append(UAV(x, y, h))
    
    def reset_state(self):
        return self.observe(self.target_loc)

    def update_state(self, engy, tar_loc, tar_dis):
        return self.observe(tar_loc)

    def target_bearing(self, target_loc=None):
        target = self.target_loc if target_loc is None else target_loc
        return math.atan2(target[1] - self.UAVs[0].y, target[0] - self.UAVs[0].x)

    def observe(self, target_loc):
        uav = self.UAVs[0]
        snr_span = self.a2g_snr_max - self.a2g_snr_min
        s = np.zeros(self.state_dim)
        s[0] = uav.x / self.length
        s[1] = uav.y / self.width
        # Target relative to the UAV, so one heading rule generalizes across legs.
        s[2] = (target_loc[0] - uav.x) / self.length
        s[3] = (target_loc[1] - uav.y) / self.width
        s[4] = self.data_size / self.max_payload
        s[5] = self.out_time / self.outage_max_s
        s[6] = np.clip((self.current_a2g_snr - self.a2g_snr_min) / snr_span, 0.0, 1.0)
        s[7] = np.clip(self.current_g2a_outage_probability, 0.0, 1.0)
        if self.map_neighborhood:
            # Ring of one-step candidates, starting at the target bearing and turning counter-clockwise.
            bearing = self.target_bearing(target_loc)
            for k in range(self.NEIGHBOR_DIRECTIONS):
                angle = bearing + 2 * math.pi * k / self.NEIGHBOR_DIRECTIONS
                point = np.array([[uav.x + self.dist_max * math.cos(angle), uav.y + self.dist_max * math.sin(angle)]])
                point = np.clip(point, [self.min_x, self.min_y], [self.max_x, self.max_y])
                s[8 + k] = np.clip((self.get_date_rate_A2G(point) - self.a2g_snr_min) / snr_span, 0.0, 1.0)
                s[8 + self.NEIGHBOR_DIRECTIONS + k] = np.clip(self.get_g2a_outage_probability(point), 0.0, 1.0)
        return s

    def action_to_absolute(self, action):
        """Convert a policy action to the absolute [heading, distance] World executes."""
        heading, distance = float(action[0]), float(action[1])
        if self.relative_heading:
            heading = self.target_bearing() + heading
        return np.array([math.atan2(math.sin(heading), math.cos(heading)), distance])

    def absolute_to_action(self, absolute):
        """Inverse of action_to_absolute; used to label demonstrations in policy coordinates."""
        heading, distance = float(absolute[0]), float(absolute[1])
        if self.relative_heading:
            heading -= self.target_bearing()
        return np.array([math.atan2(math.sin(heading), math.cos(heading)), distance])

    def get_reward(self, fa, dis_forward, pending_upload):
        """Per-slot shaping in seconds: -T and -tau, plus progress and boundary terms.

        T_out is a hard constraint enforced by termination, not a per-slot cost.
        """
        speed = self.dist_max / self.delta_T
        terms = {
            "time": -self.delta_T,
            "delay": -self.delta_T if pending_upload else 0.0,
            "progress": self.PROGRESS_WEIGHT * dis_forward / speed,
            "boundary": -fa * self.BOUNDARY_SCALE,
        }
        return sum(self.add_scaled_term(name, value) for name, value in terms.items())

    def add_scaled_term(self, name, seconds):
        value = seconds * self.REWARD_SCALE
        self.add_reward_term(name, value)
        return value

    def reward_summary(self):
        return self.reward_term_totals.copy()

    @staticmethod
    def _empty_reward_terms():
        return {name: 0.0 for name in ("time", "delay", "progress", "boundary", "upload",
                                       "arrival", "completion", "incomplete_upload", "timeout", "qos")}

    def add_reward_term(self, name, value):
        self.reward_term_totals[name] = self.reward_term_totals.get(name, 0.0) + value


    def get_date_rate_A2G(self, location): #空对地
        point_km = np.asarray(location, dtype=float).reshape(-1)[:2] / 10
        return self.radio_maps.a2g_snr_db(point_km)
    
    def get_date_rate_G2A(self, location): #地对空
        point_km = np.asarray(location, dtype=float).reshape(-1)[:2] / 10
        return self.radio_maps.g2a_snr_db(point_km)

    def get_g2a_outage(self, location):
        return self.get_g2a_outage_probability(location) >= 0.5

    def get_g2a_outage_probability(self, location):
        point_km = np.asarray(location, dtype=float).reshape(-1)[:2] / 10
        return self.radio_maps.g2a_outage_probability(point_km)

    def step_inside(self, actions,s,t): # 输入动作，状态值，时间步,输出下一状态和其他标志
        fa = 0.0
        reward = 0.0
        fly_energy = 0.0
        done = False
        self.t = t+1 # 时间步+1
        actions = self.action_to_absolute(actions)
        uav_location_pre = np.zeros([self.uav_num, 2])  # make a copy of the uav's location
        uav_location = np.zeros([self.uav_num, 2]) # uav current location
        for i, uav in enumerate(self.UAVs):
            uav_location_pre[i][0] = uav.x
            uav_location_pre[i][1] = uav.y
        if len(actions) == self.uav_num * 2:
            for i, uav in enumerate(self.UAVs):
                uav.move_inside(actions[0],actions[1],self.dist_max)  # execute the action执行飞行动作
            for i, uav in enumerate(self.UAVs):
                penalty, bound = self.boundary_margin(uav) # 输出出界奖励,出界bound 为 false
                fa += penalty # 累计出界奖励Pob
                if not bound: # 如果出界
                    self.fa += 1
                    # Clip each coordinate so the UAV slides along the wall instead of
                    # repeating a cancelled move; the penalty above is still charged.
                    uav.x = float(np.clip(uav.x, self.min_x, self.max_x))
                    uav.y = float(np.clip(uav.y, self.min_y, self.max_y))
        for i, uav in enumerate(self.UAVs):
            uav_location[i][0] = uav.x
            uav_location[i][1] = uav.y
        if LA.norm(uav_location - uav_location_pre) < self.STALL_DISTANCE:
            self.stall_steps += 1
            self._stall_run += 1
            self.max_stall_run = max(self.max_stall_run, self._stall_run)
        else:
            self._stall_run = 0

        step_cur = LA.norm(uav_location - self.target_loc)  # 此位置与目标点距离
        self.target_dis = step_cur #更新目标距离
        self.min_target_distance = min(self.min_target_distance, step_cur)
                
        step_pre = LA.norm(uav_location_pre - self.target_loc)  # 前一位置与目标点距离
        step_cur_pre = LA.norm(uav_location - self.target_pre_loc) # 此位置与前一目标点距离         
        reduce_dis = step_pre - step_cur #推进距离
        
        MaxSINR_A2G = self.get_date_rate_A2G(uav_location)  # 空对地的最大dB
        MaxSINR_G2A = self.get_date_rate_G2A(uav_location)  # 地对空的最大dB
        self.current_a2g_snr = MaxSINR_A2G
        self.current_g2a_outage_probability = self.get_g2a_outage_probability(uav_location)
        g2a_outage = self.current_g2a_outage_probability >= 0.5
        if g2a_outage:
            self.out_time += self.delta_T
        self.data_rate = self.BandWidth * np.log2(1 + 10**(MaxSINR_A2G/10.0))
        self.last_uploaded_data = 0.0
        
        # Upload during this slot; tau counts every slot that starts with pending data.
        pending_upload = not self.transmit
        if pending_upload:
            self.last_uploaded_data = min(self.data_size, self.data_rate * self.delta_T)
            self.data_size -= self.last_uploaded_data
            if self.data_size <=0:
                self.data_size = 0
                self.transmit = True
                self.trans_delay_end = self.t
                self.trans_delay += self.trans_delay_end-self.trans_delay_start

        reward += self.get_reward(fa, reduce_dis, pending_upload)
        if self.last_uploaded_data > 0 and self.current_payload > 0:
            reward += self.add_scaled_term("upload", self.UPLOAD_LEG_REWARD * self.last_uploaded_data / self.current_payload)

        # A QoS violation is failure even when it coincides with endpoint arrival.
        if self.out_time > self.outage_max_s:
            reward += self.add_scaled_term("qos", -self.INFEASIBLE_PENALTY)
            self.terminal = False
            self.r = reward
            return self.update_state(fly_energy,self.target_loc,self.target_dis), reward, True, self.t, False
                
        if step_cur<=self.distance : # 达到检查点
            #只要到目标点了，就换下一个目标点，根据传输完成情况进行奖惩
            if self.transmit:  
                print("arrive and transmit complete")
                reward += self.add_scaled_term("arrival", self.ARRIVAL_REWARD)
            else:
                print("arrive but not transmit complete, data size:",self.data_size)
                reward += self.add_scaled_term("incomplete_upload", -self.INFEASIBLE_PENALTY)
                self.terminal = False
                done = True
                state_ = self.update_state(fly_energy,self.target_loc,self.target_dis)
                self.r = reward
                return state_, self.r, done, self.t, self.terminal
                     
                
            if self.Traverse_order == len(self.Traverse): #到达终点且最终数据已上传
                self.terminal = True
                done = True
                print("=================================complete task!!========================")
            else:
                self.target_pre_loc = self.target_loc #更新上一个目标点的位置
                self.data_size = self.payload_for_user(self.Traverse[self.Traverse_order])
                self.current_payload = self.data_size
                # A point with no data needs no upload, so the next leg starts clear.
                self.transmit = self.data_size <= 0
                self.trans_delay_start = self.t
                self.terminal = False
                done = False
                if self.Traverse_order < len(self.Traverse) - 1:  # 如果还有下一个目标点
                    print("=================================arrive point:",self.Traverse[self.Traverse_order])
                    self.Traverse_order += 1 #更新目标点
                    self.target = self.Users[self.Traverse[self.Traverse_order]-1]  # 更新目标点
                    self.target_loc = np.array([self.target.x, self.target.y])
                else:
                    print("=================================arrive point:",self.Traverse[self.Traverse_order])
                    self.Traverse_order += 1 #更新目标点
                    self.target_loc = self.end_loc
        else:
            self.terminal = False
            done = False

        if self.terminal:
            print("完成任务所用时间为：",self.t*self.delta_T)
            reward += self.add_scaled_term("completion", self.COMPLETION_REWARD)
        elif self.t >= self.T:
            done = True
            reward += self.add_scaled_term("timeout", -self.INFEASIBLE_PENALTY)
            
        state_ = self.update_state(fly_energy,self.target_loc,self.target_dis)  # 进入下一状态
        self.r = reward
        
 
        return state_, self.r,done,self.t,self.terminal

    #计算出界惩罚fa=1
    def boundary_margin(self, uav):
        """检查无人机是否出界并计算惩罚"""
        margin = 1.0  # 边界容忍度
        penalty_factor = 100 / self.uav_num
        
        # 计算到边界中心的距离
        center_x = (self.max_x + self.min_x) / 2
        center_y = (self.max_y + self.min_y) / 2
        
        # 计算超出边界的距离
        x_exceed = max(abs(uav.x - center_x) - margin * (self.max_x - self.min_x) / 2, 0.0)
        y_exceed = max(abs(uav.y - center_y) - margin * (self.max_y - self.min_y) / 2, 0.0)
        
        if x_exceed == 0.0 and y_exceed == 0.0:
            return 0.0, True  # 无惩罚，在边界内
        else:
            penalty = penalty_factor * (x_exceed**2 + y_exceed**2)
            return penalty, False  # 有惩罚，出界

    # 推进能量
    # def power(self,v):
    #     P0 = 79.8563
    #     Pi = 88.6279
    #     U_tip = 120
    #     v0 = 4.03
    #     d1 = 0.6
    #     s = 0.05
    #     rho = 1.225
    #     A = 0.503
    #     P_h = P0 * (1 + 3 * v ** 2 / U_tip ** 2) + Pi * math.sqrt(
    #         math.sqrt(1 + v ** 4 / (4 * v0 ** 4)) - v ** 2 / (2 * v0 ** 2)) + 0.5 * d1 * rho * s * A * v ** 3
    #     return P_h
