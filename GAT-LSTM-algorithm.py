# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:58:57 2023

@author: ZF
"""

# -*- coding: utf-8 -*-

import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import matplotlib.pyplot as plt
import time
from collections import namedtuple, deque
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.nn import GATConv
import torch.nn.utils as nn_utils
import math
import pandas as pd
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops
import collections
from collections import defaultdict

# 添加随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


#1. 数据加载函数
def load_distance_matrix(file_name="distance_matrix4.txt"):
    # 原始的加载逻辑
    raw_distance_matrix = np.loadtxt(file_name, delimiter=' ', encoding='utf-8')
    
    # 创建一个新的 matrix，行和列都比原 matrix 多1
    new_distance_matrix = np.zeros((raw_distance_matrix.shape[0] + 1, raw_distance_matrix.shape[1] + 1))
    
    # 将原 matrix 的数据填入新 matrix 的从1开始的位置
    new_distance_matrix[1:, 1:] = raw_distance_matrix
    
    return new_distance_matrix

def load_demand_data_single_day(file_path, day_index):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    
    demand_data = [
        (
            float(line.split('\t')[0]),  # Time
            line.split('\t')[1],  # Vehicle type
            int(line.split('\t')[2]),  # Start location
            int(line.split('\t')[3]),  # End location
            float(line.split('\t')[4].strip()),  # SOC
            day_index,  # Day of the week
            int((float(line.split('\t')[0]) * 60) // 15)  # 15-min time slot index
        )
        for line in data[1:]
    ]
    
    # Sort the data by time to ensure the correct calculation of future demand
    demand_data.sort(key=lambda x: x[0])
    return demand_data

# Define file paths and day mappings
file_paths = ["demand_data.txt", "demand_data1.txt", "demand_data2.txt", "demand_data3.txt", "demand_data4.txt", 
              "demand_data5.txt", "demand_data6.txt"]

day_mapping = {
    "demand_data.txt": 0, "demand_data1.txt": 1, "demand_data2.txt": 2, 
    "demand_data3.txt": 3, "demand_data4.txt": 4, "demand_data5.txt": 5, 
    "demand_data6.txt": 6
}

# Load and combine all data
all_data = []
for file_path in file_paths:
    all_data.extend(load_demand_data_single_day(file_path, day_mapping[file_path]))

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(all_data, columns=['Time', 'Vehicle_Type', 'Start_Location', 'End_Location', 'SOC', 'Day_of_Week', 'Time_Slot'])

def calculate_future_demand(demand_data):
    time_window = 15 / 60  # 15分钟转换为小时
    future_demand_dict = defaultdict(int)
    demand_times = [data[0] for data in demand_data]

    n = len(demand_times)
    j = 0  # 初始化第二个指针

    for i, time_point in enumerate(demand_times):
        # 移动第二个指针，直到找到一个时间点不在当前时间点的15分钟窗口内
        while j < n and demand_times[j] < time_point + time_window:
            j += 1
        
        # 计算在15分钟窗口内的换电需求数
        future_demand_count = j - i - 1  # 减1是因为我们不包括当前时间点
        future_demand_dict[time_point] = future_demand_count

    return future_demand_dict


# 2.环境定义
# 检查GPU的可用性并定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch_geometric.data import Data

class Environment:
    def __init__(self):
        self.all_demand_data = [load_demand_data_single_day(file_path, day_mapping[file_path]) for file_path in file_paths]
        self.distance_matrix = load_distance_matrix()  # 加载distance_matrix
        
        # 电站节点位置数据
        self.stations = [
            4, 17, 11, 23, 36, 41, 43, 46, 32, 53,
            64, 66, 61, 70, 72, 79, 80, 89, 92, 95,
            98, 96, 103, 121, 119, 12, 30, 48, 67, 86,
            112, 117, 137, 142, 169, 127, 107, 130, 131, 139,
            145, 148, 152, 156, 158, 162, 165, 144, 124, 170,
            172, 175, 178, 183, 189, 196, 155, 197, 202, 102
        ]
        
        # Initialize the battery status: 1 represents full, 0 represents charging
        self.batteries_status = {station: [1 for _ in range(10)] for station in self.stations}
        
        # Predicted full charge time for each battery, initialize with -1 indicating no ongoing charging process
        self.batteries_full_time = {station: [0 for _ in range(10)] for station in self.stations}
        
        self.demand_data = []  # 初始化 demand_data 为空列表
        # Initializing car_needs and car_times as empty dictionaries
        self.current_demand_index = 0  # 在此初始化 current_demand_index
        # 在此处初始化要跟踪的值
        self.cumulative_detour_time = 0
        self.cumulative_queue_time = 0
        self.exchange_failures = 0

        self.is_last_epoch = False
        self.exchange_failed_flag = False  # 在__init__方法中初始化属性
        self.epoch_cumulative_reward = 0
        
        self.epoch_cumulative_detour_time = 0
        self.epoch_cumulative_queue_time = 0
        self.epoch_cumulative_exchange_failures = 0
        self.episode_cumulative_reward = 0
        
        self.current_epoch_episode_count = 0
        self.car_charging_details_ch = []  # 用于记录换电详情

    
    def set_last_epoch(self, status):
        self.is_last_epoch = status
    
    def reset_day(self):
        """Reset all batteries to full charge at the start of each day."""
        for station in self.stations:
            self.batteries_status[station] = [1 for _ in range(10)]  # Reset to full charge
            self.batteries_full_time[station] = [0 for _ in range(10)]  # Reset full charge time to -1 indicating no ongoing charging process
        
    def reset(self):
        """ Reset the environment for a new episode """
        self.current_demand_index = 0  # 创建一个新的索引来跟踪当前的需求
        print(f"Number of sorted demands: {len(self.demand_data)}")  # 打印排序后的需求数量
        # 在每个新的episode开始时重置跟踪的值
        self.cumulative_detour_time = 0
        self.cumulative_queue_time = 0
        self.exchange_failures = 0
  
    def load_demand_data(self, demand_data):
        """Load demand data for the current day."""
        self.reset_day()  # Reset battery statuses at the start of each day
        # Sort the demand data based on time before processing it
        self.demand_data = sorted(demand_data, key=lambda x: x[0])
        self.future_demand_dict = calculate_future_demand(self.demand_data)
  
    def next_car(self):
        """ Get the next demand from the sorted demand data """
        if self.current_demand_index < len(self.demand_data):
            demand = self.demand_data[self.current_demand_index]
            self.current_demand_index += 1
            return demand
        return None
        
    def step(self, action):
       
        recommended_station = self.stations[action]
        
        done = False  # 在方法开始时初始化 done 变量
        # Get the current car ID
        car = self.next_car()
        if car is None:
            
            # Update the epoch cumulative values
            self.epoch_cumulative_reward += self.episode_cumulative_reward
            self.epoch_cumulative_detour_time += self.cumulative_detour_time
            self.epoch_cumulative_queue_time += self.cumulative_queue_time
            self.epoch_cumulative_exchange_failures += self.exchange_failures
    
            # Reset episode cumulative values for the next episode
            self.episode_cumulative_reward = 0
            self.cumulative_detour_time = 0
            self.cumulative_queue_time = 0
            self.exchange_failures = 0
            
            # Update the current episode count for the epoch
            self.current_epoch_episode_count += 1
            
             # Check if the episode is done (no more cars)
            done = self.current_demand_index >= len(self.demand_data)
            
            if done and self.is_last_epoch:
                
                # 计算和打印 epoch 的平均统计信息
                avg_epoch_reward = self.epoch_cumulative_reward / self.current_epoch_episode_count
                avg_epoch_detour_time = self.epoch_cumulative_detour_time / self.current_epoch_episode_count
                avg_epoch_queue_time = self.epoch_cumulative_queue_time / self.current_epoch_episode_count
                avg_epoch_exchange_failures = self.epoch_cumulative_exchange_failures / self.current_epoch_episode_count
                
                # 打印epoch的平均统计信息
                print(f"Average Epoch Reward: {avg_epoch_reward}")
                print(f"Average Epoch Detour Time: {avg_epoch_detour_time}")
                print(f"Average Epoch Queue Time: {avg_epoch_queue_time}")
                print(f"Average Epoch Exchange Failures: {avg_epoch_exchange_failures}")
                
                # Reset the epoch cumulative values for the next epoch
                self.epoch_cumulative_reward = 0
                self.epoch_cumulative_detour_time = 0
                self.epoch_cumulative_queue_time = 0
                self.epoch_cumulative_exchange_failures = 0
        
                # Reset the epoch episode count for the next epoch
                self.current_epoch_episode_count = 0
            
            return None, 0, True, {}  # No more cars, episode is done

        # Reset the exchange_failed_flag for the new car
        self.exchange_failed_flag = False
        
        current_time = car[0]  # Getting current time directly from car parameter
        
        # Step 1: Update the current state of the battery stations before the agent takes an action
        self.update_battery_statuses(current_time)
    
        # Step 2: Get the new state after updating the battery stations
        next_state = self.get_current_state(car, current_time)
        
        # Use the updated update_battery_charge method
        travel_time, queue_time, detour_time, charging_status = self.update_battery_charge(recommended_station, car, current_time)
    
        # Append the charging attempt details to car_charging_details_ch
        self.car_charging_details_ch.append({
            'Car_ID': car,
            'Start_Location': car[2],
            'End_Location': car[3],
            'Station_ID': recommended_station,
            'Travel_Time': travel_time,
            'Queue_Time': queue_time,
            'Charging_Status': charging_status
        })

        # Step 3: Calculate the reward based on the current state and the action the agent is about to take
        #reward = self.get_reward(car, recommended_station)
        reward, detour_time, queue_time = self.get_reward(car, recommended_station)
        
        # Update the cumulative reward for the episode
        self.episode_cumulative_reward += reward
        
        # 在step函数中更新跟踪的值
        self.cumulative_detour_time += detour_time
        self.cumulative_queue_time += queue_time
          
        return next_state, reward, done, {}        
    
    def is_peak_hour(self, time):
        """判断给定的时间是否为峰值时段"""
        hours = int(time)
        if (9 <= hours < 12) or (16 <= hours < 19):
            return 1
        return 0
    
    #每个时间步之前检查状态特征是否更新
    def update_battery_statuses(self, current_time):
        """ Update the statuses of all batteries based on the current time """
        for station in self.stations:
            for i in range(10):  # Assuming there are 10 batteries at each station
                full_time = self.batteries_full_time[station][i]
                
                # If the battery is charging and the current time is past the predicted full time
                if self.batteries_status[station][i] == 0:
                    if full_time > 0 and current_time >= full_time:
                        self.batteries_status[station][i] = 1  # Update status to full
                        self.batteries_full_time[station][i] = 0  # Reset full time to 0
                    # elif full_time > 0 and current_time < full_time:
                    #     pass
                    elif full_time == -1:
                        # If the battery cannot be charged within the day, disable it for the day
                        self.batteries_status[station][i] = -1  # Disable the battery for the day       
                # If the battery is full and not charging
                elif self.batteries_status[station][i] == 1:
                    self.batteries_full_time[station][i] = 0  # Ensure full time is 0 for full batteries not charging
                # If the battery cannot be charged within the day, we ensure that the full_time is -1
                elif self.batteries_status[station][i] == -1:
                    self.batteries_full_time[station][i] = -1  # Ensure full time is -1 for batteries that cannot be charged today
    
    
    #当前的状态特征获取                
    def get_current_state(self, car,current_time):
        #state_vector=[]
        
        # First, update the battery statuses based on the current time
        self.update_battery_statuses(current_time)   
        
        # Node features
        node_features = []
        
        # Get the initial position of the car
        car_pos1 = car[2] / 215.0  # 归一化到0-1范围
       
        # 获取未来15分钟的换电需求
        future_demand1 = self.future_demand_dict.get(car[0], 0) / 100  # 使用最大可能值归一化
        
        # Get the actual time point (in hours) for the current car
        current_time_point1 = car[0] / 24.0  # 归一化到0-1范围
        
        # Get the destination location
        destination_location = car[3] / 215.0  # 归一化到0-1范围
        
        # 为当前的电动车添加特征
        car_features = [current_time_point1, car[4], float(self.is_peak_hour(car[0])), car_pos1, destination_location, future_demand1]

        min_id = min(self.stations)
        max_id = max(self.stations)

        # 每个换电站的节点特征可以是电池状态和充满电的时间
        for station in self.stations:
            station_features = []
            
            # 1. 添加换电站的编号作为特征（归一化）
            normalized_station_id = (station - min_id) / (max_id - min_id)
            station_features.append(normalized_station_id)
            
            # 2.计算电站的满电电池数量
            full_battery_count = sum([1 for status in self.batteries_status[station] if status == 1])
            station_features.append(full_battery_count / 10)  # 归一化
            
            # 3 & 4. 计算正在充电的电池的平均和最小预计剩余充电时间
            charging_times_left = [max(0, full_time - current_time) for full_time in self.batteries_full_time[station] if full_time > current_time]
            if charging_times_left:
                avg_charging_time_left = sum(charging_times_left) / len(charging_times_left) / 5
                min_charging_time_left = min(charging_times_left) / 5
            else:
                avg_charging_time_left = 0  # 用0表示所有电池都已充满电
                min_charging_time_left = 0  # 用0表示所有电池都已充满电
            
            station_features.append(avg_charging_time_left)
            station_features.append(min_charging_time_left)

                    
            # 5.计算电站的繁忙程度和电池的充电状态
            busy_level = sum([1 for status in self.batteries_status[station] if status == 0 or status == -1]) / 10
            station_features.append(busy_level)    
            
            # 6. 计算无法使用的电池比例（状态为-1的电池比例）
            unavailable_battery_ratio = sum([1 for status in self.batteries_status[station] if status == -1]) /10
            station_features.append(unavailable_battery_ratio)
            
            #state_vector.append(station_features)   
            
            node_features.append(station_features)
        
        # Ensure the car features length is the same as station features length
        car_features.extend([0] * (len(station_features) - len(car_features)))
    
        # Add car features to state_vector
        #state_vector.append(car_features)
        node_features.append(car_features)
        
        # Edge index and attributes
        edge_index = []
        edge_attributes = []
        car_idx = len(node_features) - 1  # Car node index should be the last one
        for station_idx in range(len(self.stations)):
            edge_index.append([car_idx, station_idx])
            edge_attributes.append(self.distance_matrix[car[2]][self.stations[station_idx]])
            
            edge_index.append([station_idx, car_idx])
            edge_attributes.append(self.distance_matrix[self.stations[station_idx]][car[2]])  # Reverse the distance for the opposite edge

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attributes = torch.tensor(edge_attributes, dtype=torch.float32)

        # Create graph data
        graph_data = Data(x=torch.tensor(node_features, dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attributes)

        return graph_data.to(device)
    
        # # Convert state_vector to a tensor and add extra dimensions to create a 4D tensor
        # state_tensor = torch.tensor(state_vector, dtype=torch.float32).to(device)
        # state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
        # state_tensor = state_tensor.unsqueeze(0)  # Add channel dimension
        
        # return state_tensor

    def calculate_travel_time(self, start, end):
            """计算从start到end的行驶时间"""
            distance = self.distance_matrix[start][end] / 1000
            return distance / 18         
        
    def update_battery_charge(self, recommended_station, car,current_time):
        # Initialize charging_status
        charging_status = "Success"  # Default to Success, change to "Failed" if needed
        queue_time = 0  # Initialize queue_time
        detour_time = 0  # Initialize detour_time
        travel_time = 0  # Initialize travel_time
      
        # # Step 1: Update the battery statuses based on the current time
        self.update_battery_statuses(current_time)
        
        # Step 2: Get the car position and the state of charge (SOC)
        car_pos =car[2]  
        car_soc = car[4] 
        destination_location =car[3]

        # 首先查找一个充满的电池
        locked_battery_idx = None
        for i in range(10):
            if self.batteries_status[recommended_station][i] == 1:
                self.batteries_status[recommended_station][i] = 0  # Set to charging
                locked_battery_idx = i
                break
        if locked_battery_idx is None:
             # 检查是否所有电池都无法在当天充满
            if all(x == -1 for x in self.batteries_full_time[recommended_station]):
                # 你可以选择在这里返回一个特殊标记或抛出一个异常来表示换电失败
                charging_status = "Failed"
                if not self.exchange_failed_flag:
                    self.exchange_failures += 1  # 增加换电失败的计数器
                    self.exchange_failed_flag = True  # 设置标志以防止多次增加失败计数
                    # Record the failure in car_charging_details_ch
                    self.car_charging_details_ch.append({
                        'Car_ID': car,
                        'Start_Location': car_pos,
                        'End_Location': destination_location,
                        'Station_ID': recommended_station,
                        'Travel_Time': None,
                        'Queue_Time': None,
                        'Charging_Status': 'Failed'
                    })
                # Return early since charging failed
            #    return None, None, None, charging_status
                return travel_time, queue_time, detour_time, charging_status

            else:
                # 如果没有充满的电池，为该车辆锁定预计充电时间最短的电池
                earliest_full_time = min(filter(lambda x: 0< x < 24, self.batteries_full_time[recommended_station]))
                locked_battery_idx = self.batteries_full_time[recommended_station].index(earliest_full_time)
                # 设置一个高值表示这块电池已被锁定，防止被其他电动车锁定
                self.batteries_full_time[recommended_station][locked_battery_idx] =  float('inf') 
    
        ## Step 4: Calculate the travel time and the charging duration        
        travel_time = self.calculate_travel_time(car_pos,recommended_station)
    
        # Calculate SOC when starting to charge
        start_soc = car_soc - (self.distance_matrix[car_pos][recommended_station] / 1000 * 1.66667 / 100)

        charge_duration = (1 - start_soc)*100* 3 / 60  # Convert minutes to hours
    
        # Convert current_timefrom HH.MM format to decimal hours
        hours = int(current_time)
        minutes = (current_time - hours) * 60
        decimal_time = hours + (minutes / 60)
    
        # # Step 5: Calculate the predicted arrival time at the station and the full charge time
        arrival_time = decimal_time + travel_time
             
        # Calculate the predicted full charge time
        full_time = arrival_time + charge_duration
        if full_time >= 24:
            charging_status = "Failed"
            full_time = -1
            self.batteries_status[recommended_station][locked_battery_idx] = -1
        else:
            self.batteries_full_time[recommended_station][locked_battery_idx] = full_time
            self.batteries_status[recommended_station][locked_battery_idx] = 0
    
        # Assuming travel_time, queue_time, and detour_time are calculated within this method:
        return travel_time, queue_time, detour_time, charging_status
        
    
    #奖励函数
    def get_reward(self, car, recommended_station):      
        MAX_TOTAL_TIME = 3 # 最大可能时间设置为1小时
        PENALTY_FOR_FAILURE = 5  # 设置一个负奖励值表示换电失败
        # Calculate detour time
        
        # 从car参数中提取必要的信息
        car_pos = car[2]
        destination_location = car[3]
        current_time = car[0]
        
        # 计算从初始位置到推荐换电站的时间
        time_to_station = self.calculate_travel_time(car_pos, recommended_station)
        # 计算从推荐换电站到目的地的时间
        time_from_station_to_destination = self.calculate_travel_time(recommended_station, destination_location)
        # 计算从初始位置直接到目的地的时间
        direct_time_to_destination = self.calculate_travel_time(car_pos, destination_location)
        # 计算绕行时间
        detour_time = time_to_station + time_from_station_to_destination - direct_time_to_destination
        
        # Introducing coefficients for the detour and queue time
        coefficient_detour = 0.3
        coefficient_queue = 0.7
        
        # Introducing new factors to consider the station's busyness level and battery charging status
        busy_penalty_factor = 0.3  # 可以调整此值来改变繁忙惩罚的影响
        charging_penalty_factor = 0.3  # 可以调整此值来改变充电惩罚的影响
        peak_time_penalty_factor = 0.3
        future_demand_reward_factor = 0.3
        
        # 计算电站的繁忙程度和电池的充电状态
        busy_level = sum([1 for status in self.batteries_status[recommended_station] if status == 0 or status == -1]) / 10
        charging_status = sum(max(0, full_time - current_time) for full_time in self.batteries_full_time[recommended_station] if full_time > 0 and full_time != float('inf')) / (10*5)#一块电池的最长充电时间为5h

        # 计算需求压力因子（结合高峰时段和未来换电需求）
        future_demand = self.future_demand_dict.get(car[0], 0) / 100
        demand_pressure_factor = (self.is_peak_hour(current_time) * peak_time_penalty_factor) + (future_demand * future_demand_reward_factor)

        # Calculate the predicted arrival time at the station
        # 将当前时间从HH.MM格式转换为小数小时格式
        hours = int(current_time)
        minutes = (current_time - hours) * 60
        decimal_time = hours + (minutes / 60)
    
        arrival_time = decimal_time + time_to_station

        # 试图更新电池充电状态并获取返回值
        update_result = self.update_battery_charge(recommended_station, car, current_time)
        
        # 如果更新结果表明换电失败，则返回负奖励
        if update_result == "换电失败":
            # 计算排队时间为24-电动车的到达时刻
            queue_time = 24 - arrival_time
            # 计算总时间和奖励
            # 确保队列时间为正数
            queue_time = max(0, queue_time)
            total_time = coefficient_detour * detour_time + coefficient_queue * queue_time 
            reward =MAX_TOTAL_TIME - total_time  # 添加额外的惩罚
            #reward = - total_time  # 添加额外的惩罚
            reward -= PENALTY_FOR_FAILURE
            reward += busy_penalty_factor * (1-busy_level)
            reward += charging_penalty_factor * (1-charging_status)
            #reward -= demand_pressure_factor * (detour_time + queue_time)
            return reward, detour_time, queue_time
        
        # Calculate queue time based on the battery's predicted full charge time
        full_times_after_arrival = [time for time in self.batteries_full_time[recommended_station] if time >= arrival_time and time != float('inf')]

        # Check if all batteries are charging
        all_batteries_charging = not any([status == 1 for status in self.batteries_status[recommended_station]])

        if all_batteries_charging and full_times_after_arrival:  # If no batteries are full and there are predicted times after arrival
            earliest_full_time = min(full_times_after_arrival)
            queue_time = earliest_full_time - arrival_time
        else:
            queue_time = 0

        total_time = coefficient_detour * detour_time + coefficient_queue * queue_time 
        reward =   MAX_TOTAL_TIME- total_time

        reward += busy_penalty_factor *(1- busy_level)
        reward += charging_penalty_factor *(1- charging_status)
       
        return reward , detour_time, queue_time

#3.图神经网络训练    
# 3. DQN Network with Dueling Architecture
class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.3):
        super(DuelingDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout = nn.Dropout(p=dropout_rate)  # Adding dropout
        
        # Dueling branches
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            self.dropout,  # Adding dropout after the ReLU
            nn.Linear(32, 1)  # Outputs a single value
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            self.dropout,  # Adding dropout after the ReLU
            nn.Linear(32, output_size)  # Outputs advantage for each action
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s, a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values    

# GAT Network
class GATNetwork(nn.Module):
    def __init__(self, node_feature_dim, output_size):
        super(GATNetwork, self).__init__()
        
        # 1. 叠加多个 GAT 层
        self.conv1 = GATConv(node_feature_dim, 32, heads=2)  # Use 2 attention heads
        self.batch_norm1 = nn.BatchNorm1d(32 * 2)  # Adding batch normalization
        self.conv2 = GATConv(32 * 2, 64, heads=2)  # Added another GAT layer
        self.batch_norm2 = nn.BatchNorm1d(64 * 2)  # Adding batch normalization
        
        self.fc = nn.Linear(64 * 2, output_size)  # 64 features times 2 heads
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # 4. 使用不同的激活函数
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.dropout(x)  # Apply dropout to input features
        
        # Using LeakyReLU activation
        x = self.leaky_relu(self.conv1(x, edge_index))
        x = self.batch_norm1(x)  # Applying batch normalization
        x = self.leaky_relu(self.conv2(x, edge_index))
        x = self.batch_norm2(x)  # Applying batch normalization
        
        x = self.fc(x)
        return x[-1]  # Return only the car's feature vector   

# Combined GAT-DuelingDQN Model

class CombinedDuelingModel(nn.Module):
    def __init__(self, node_feature_dim, dqn_output_size):
        super(CombinedDuelingModel, self).__init__()
        self.gat = GATNetwork(node_feature_dim, 64)
        self.dqn = DuelingDQNNetwork(64, dqn_output_size)

    def forward(self, data):
        x = self.gat(data)
        return self.dqn(x.unsqueeze(0))    
              
# 初始化函数
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Initialize the model
dueling_combined_model = CombinedDuelingModel(node_feature_dim=6, dqn_output_size=60).to(device)
dueling_combined_model.apply(weights_init)


##4. 优先经验回放
class PriorityReplayBuffer:
    
    def __init__(self, capacity=10000, alpha=0.6, n_step=4, gamma=0.99, batch_size=64, q_network=None):
        self.alpha = alpha
        self.capacity = capacity
        self.experiences = []
        self.priorities = torch.zeros(capacity).to(device)
        self.batch_size = batch_size
        self.position = 0
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)  # 新增n步缓冲区       
        self.q_network = q_network
        
    def add(self, experience, priority):
        """Add an experience to the buffer."""
        self.n_step_buffer.append(experience)
        
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # Check if any reward in the n_step buffer is None and print a warning if so
        for i, exp in enumerate(self.n_step_buffer):
            if exp[2] is None:
                print(f"Experience at index {i} has None as reward: {exp}")
        
        # Compute total reward over n steps
        total_reward = sum([(self.gamma**i) * exp[2] for i, exp in enumerate(self.n_step_buffer)])
        #total_reward = experience[2]
        # Get the final state from the n-step buffer
        final_state = self.n_step_buffer[-1][3]

        # If the episode has not ended, add the estimated value of the final state
        # if final_state is not None:
        #     with torch.no_grad():
        #         final_state_value = self.q_network(final_state).max().item()
        #         total_reward += (self.gamma ** self.n_step) * final_state_value
      
        #state, action, _, last_next_state = self.n_step_buffer[0]
        state, action, _, last_next_state, _ = self.n_step_buffer[-1]
        new_priority = torch.abs(torch.tensor(total_reward)).to(device)  # Compute new priority based on the total reward
        self._add((state, action, total_reward, last_next_state), new_priority)
          
    def _add(self, experience, priority):
        """Private method to add experience and priority to the buffer."""
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
        else:
            self.experiences[self.position] = experience
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity    
     
    def sample(self, batch_size, beta=0.4):
        """Sample a batch of experiences based on their priorities."""
        if len(self.experiences) == self.capacity:
            priors = self.priorities
        else:
            #priors = self.priorities[:self.position]
            priors = self.priorities[:len(self.experiences)]

        # Ensure all priorities are positive
        priors = torch.clamp(priors, min=1e-10) 
        probs = priors.pow(self.alpha)+ 1e-10

        # Normalize probabilities so they sum to 1
        probs /= probs.sum()
        
        indices = torch.multinomial(probs, batch_size, replacement=True)
        samples = [self.experiences[idx] for idx in indices.cpu().numpy()]
        total = len(self.experiences)
        weights = (total * probs[indices]).pow(-beta)
        weights /= weights.max()
        weights = weights.to(device)

        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            assert priority > 0, "Priority should be positive"
            assert not np.isinf(priority), "Inf values found in new priorities during update"
            assert not np.isnan(priority), "NaN values found in new priorities during update"
    
            self.priorities[index] = torch.tensor(priority, device=self.priorities.device)  # Modify this line

    def __len__(self):
        return len(self.experiences)      
  
     
# 5. train_DQN训练过程
scaler = GradScaler()

def update_replay_buffer(replay_buffer, batch, device, gamma, policy_model, target_model, criterion, optimizer, scaler):
    states, actions, rewards, next_states, indices, weights = zip(*batch)
    states = [s.to(device) for s in states]
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    #next_states = [s.to(device) for s in next_states]
    next_states = [s.to(device) if s is not None else None for s in next_states]


    with autocast():
        q_values_list = [policy_model(state) for state in states]
        q_values = torch.cat(q_values_list, dim=0).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        next_q_values_list = []
  
        for state in next_states:
            if state is not None:
                next_q_values_list.append(torch.max(target_model(state)))
            else:
                next_q_values_list.append(torch.tensor(0.0).to(device))  # Assuming a reward of 0.0 for terminal state
        next_q_values = torch.stack(next_q_values_list).detach()

        target_q_values = rewards + gamma * next_q_values#目标q值
        assert not torch.isnan(target_q_values).any(), "NaN values found in target_q_values"
        assert not torch.isinf(target_q_values).any(), "Inf values found in target_q_values"
       
        weights = torch.tensor(weights, dtype=torch.float32).to(device)  # Convert weights to a tensor
        loss = criterion(q_values, target_q_values)
        loss = (loss * weights).mean()  # Adjust loss with weights
       
    td_error = target_q_values - q_values
    
    priorities = (torch.abs(td_error) + 1e-5).detach().to(device)
    
    replay_buffer.update_priorities(indices, priorities.cpu().numpy())
    return loss, priorities


# 训练的循环部分:
def initialize_training(env, file_paths, day_mapping, policy_model, epsilon_start, device):  
    start_time = time.time()
    episode_rewards = []
    episode_losses = []
    epsilon = epsilon_start
    replay_buffer = PriorityReplayBuffer(capacity=10000, alpha=0.6, n_step=4, gamma=0.99, batch_size=64, 
                                         q_network=policy_model)
    overall_step = 0
    all_demand_data = [load_demand_data_single_day(file_path, day_mapping[file_path]) for file_path in file_paths]
    return start_time, episode_rewards, episode_losses, epsilon, replay_buffer, overall_step, all_demand_data

def handle_and_add_experience(car, env, policy_model, epsilon, device, replay_buffer, batch, total_reward, step_counter, target_update_steps, target_model):
    if car is None:
        return None, 0   
    state = env.get_current_state(car,car[0])  
    state = state.clone().detach().to(device)

    with autocast():
        q_values = policy_model(state).squeeze(0)

    if random.uniform(0, 1) < epsilon:
        action = random.choice(range(60))
    else:
        action = torch.argmax(q_values).item() %60  

    # Use the step method to apply the action and get the new state and reward
    next_state, reward, done, _ = env.step(action)
         
    #Set the initial priority to the maximum priority in the replay buffer
    if len(replay_buffer) > 0:
        initial_priority = replay_buffer.priorities.max().item()
    else:
        initial_priority = 1  # Set to 1 if the replay buffer is empty

    if next_state is not None:  
        next_state = next_state.clone().detach().to(device)

    experience = [state, action, reward, next_state, initial_priority]
    
    replay_buffer.add(experience, initial_priority)  
    batch.append(experience)  
    
    step_counter += 1  # 更新步数计数器


    return experience, reward, step_counter
      
def train_one_day(day_data, env, policy_model, epsilon, device, 
                  replay_buffer, batch_size, gamma, target_model, criterion, optimizer, scaler, batch, step_counter, target_update_steps):
    env.load_demand_data(day_data)
    env.reset()
    car = env.next_car()

    #batch = []  # 初始化批处理列表
    total_reward = 0
    total_loss = 0
    num_loss_updates = 0
    step=0
    optimizer.zero_grad()
    
    while car or len(batch) > 0:
        step+=1
        experience, reward, step_counter = handle_and_add_experience(car, env, policy_model, epsilon, device, 
                                                       replay_buffer, batch, total_reward, step_counter, 
                                                                 target_update_steps, target_model)
        
        total_reward += reward
        
        car = env.next_car()
        if step<10 and len(replay_buffer.experiences) >= batch_size or (not car and len(batch) > 0):
            samples, indices, weights = replay_buffer.sample(min(len(replay_buffer.experiences), batch_size))
            batch = [sample + (index, weight) for sample, index, weight in zip(samples, indices, weights)]
  
            if batch:
                loss, priorities = update_replay_buffer(replay_buffer, batch, device, gamma, 
                                                        policy_model, target_model, criterion, optimizer, scaler)
                total_loss += loss.item()
                num_loss_updates += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scaler.scale(loss).backward()
                #replay_buffer.update_priorities(indices, priorities)
                replay_buffer.update_priorities(indices, priorities.cpu().numpy())
    
                nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1)  # 将此行移到scaler.unscale_之前
                # scaler.unscale_(optimizer)
                # scaler.step(optimizer)
                # scaler.update()
                # optimizer.zero_grad()
                if step_counter % target_update_steps == 0:
                    tau = 0.01
                    for target_param, local_param in zip(target_model.parameters(), policy_model.parameters()):
                        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            batch.clear()
    print(total_reward,'   ',total_loss)
    
    # 在函数的末尾，保存car_charging_details_ch到CSV文件
    df = pd.DataFrame(env.car_charging_details_ch)
    df.to_csv('D:/path/to/your/directory/car_charging_details_ch.csv', index=False)

#    df.to_csv('/mnt/data/car_charging_details_ch.csv', index=False)
    env.car_charging_details_ch = []  # 重置列表以供下一天使用
    
    
    return total_loss, num_loss_updates, total_reward,batch, step_counter
    
def train_double_dqn_multi_step(policy_model, target_model, optimizer, criterion, env, scheduler, file_paths, 
                                day_mapping, epsilon_start, device, scaler, gamma=0.99, 
                                epsilon_end=0.0, epsilon_decay=0.8,  batch_size=64):
    start_time, episode_rewards, episode_losses, epsilon, replay_buffer, overall_step, all_demand_data = initialize_training(env, file_paths, day_mapping, policy_model, epsilon_start, device)
    
    num_epochs  =1
    target_update_steps =100# 新增超参数，每50步更新一次目标网络
    step_counter = 0  # 计步器
    batch = []  # 初始化一个空的batch列表
    
    for epoch in range(num_epochs):  # 这里将外层循环定义为epoch
        if epoch == num_epochs - 1:  # 如果是最后一个epoch
            env.set_last_epoch(True)
        else:
            env.set_last_epoch(False)
        scheduler.step()  # 在每个epoch结束时更新scheduler
        # random.shuffle(all_demand_data)  # 随机打乱数据
        for day_data in all_demand_data:  # 这里将每天作为一个episode
            total_loss, num_loss_updates, total_reward, batch, step_counter = train_one_day(day_data, env, policy_model, epsilon,
                                                                              device, 
                                                                              replay_buffer, batch_size, gamma,
                                                                              target_model, criterion, optimizer,
                                                                              scaler, batch, step_counter, 
                                                                                        target_update_steps)
            # Compute the average loss for this episode
            if num_loss_updates != 0:
                average_loss = total_loss / num_loss_updates
            else:
                average_loss = 0
            episode_losses.append(average_loss)  # store the average loss for this episode      
            episode_rewards.append(total_reward)
            
            epsilon = max(epsilon_end, epsilon * epsilon_decay)  # reduce epsilon for the next episode

    # 训练完成后保存模型
    torch.save(policy_model.state_dict(), 'trained_model.pth')
    print("模型已保存为 trained_model.pth")


    end_time = time.time()  # 记录结束时间
    training_duration = end_time - start_time  # 计算训练总时长
    print(f"Training completed in {training_duration:.2f} seconds.")
    
    # 计算奖励的移动平均值
    moving_avg_rewards = [np.mean(episode_rewards[max(0, i-10):i+1]) for i in range(len(episode_rewards))]
    
    # 保存每个episode的总奖励、移动平均奖励和episode编号到DataFrame
    episode_data = pd.DataFrame({
        'Episode': range(1, len(episode_rewards) + 1),
        'Total Reward': episode_rewards,
        'Moving Avg Reward': moving_avg_rewards
    })
    
    # 将DataFrame保存到Excel文件
    excel_filename = 'episode_rewards3.xlsx'
    episode_data.to_excel(excel_filename, index=False)
    
    print(f"奖励数据已保存到Excel文件：{excel_filename}")

    return episode_rewards, episode_losses

            
#6.主程序
env = Environment()
# 加载第一天的数据
demand_data_first_day = load_demand_data_single_day(file_paths[0], day_mapping[file_paths[0]])
env.load_demand_data(demand_data_first_day)

# 获取第一个汽车对象和当前时间
first_car = env.next_car()
current_time = first_car[0] if first_car else 0

# Note: You'll need to update the `get_state_vector` method in your Environment class
# to return a state vector instead of a graph structure.
state_vector = env.get_current_state(first_car, current_time)
state_length = len(state_vector) if state_vector is not None else 0
num_actions = len(env.stations)

# 初始化模型
input_size = 366  # 根据新状态向量的大小来设置
output_size =60  # 根据动作空间大小来设置


combined_model = CombinedDuelingModel(node_feature_dim=6, dqn_output_size=num_actions).to(device)
policy_model = combined_model
target_model = deepcopy(combined_model)  # 使用深拷贝来创建目标模型的独立副本

# Initialize target network with the same weights as the policy network
target_model.load_state_dict(policy_model.state_dict())
target_model.eval()  # Set the target network to evaluation mode

# weight_decay 参数对应于L2正则化
optimizer = optim.Adam(policy_model.parameters(), lr=0.0001)
# 添加学习率调度器，假设每1个epochs降低学习率为原来的0.9倍
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

# 使用Huber Loss
criterion = nn.MSELoss()


rewards, losses = train_double_dqn_multi_step(policy_model, target_model, optimizer, criterion, env, scheduler, 
                                              file_paths=file_paths, day_mapping=day_mapping, epsilon_start=0.5,
                                              device=device, scaler=scaler)



#7.绘制奖励图

# plot rewards
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.show()

# # Now, you can plot the losses over episodes:
plt.figure()
plt.plot(losses)
plt.xlabel('Episode')
plt.ylabel('Average Loss')
plt.title('Average Loss per Episode')
plt.show()

# # 计算奖励的移动平均值（例如，每10个episodes）
#moving_avg_rewards = [np.mean(episode_rewards[max(0, i-10):i+1]) for i in range(len(episode_rewards))]
#moving_avg_rewards = [np.mean(rewards[max(0, i-10):i+1]) for i in range(len(rewards))]

 # 绘制原始奖励和移动平均奖励
plt.figure()
plt.plot(rewards, label='Original Rewards')
#plt.plot(moving_avg_rewards, label='Moving Average Rewards', color='orange')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.legend()
plt.show()
