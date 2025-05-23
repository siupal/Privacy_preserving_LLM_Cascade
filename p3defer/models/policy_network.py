"""
策略网络模块，实现PPO算法的策略学习框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    """策略网络，用于决策延迟动作"""
    
    def __init__(self, input_dim=96, hidden_dim=64, output_dim=3):
        """初始化策略网络
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（动作数）
        """
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        """前向传播
        
        Args:
            state: 状态向量
            
        Returns:
            动作概率分布
        """
        return self.network(state)

class ValueNetwork(nn.Module):
    """价值网络，用于评估状态价值"""
    
    def __init__(self, input_dim=96, hidden_dim=64):
        """初始化价值网络
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """前向传播
        
        Args:
            state: 状态向量
            
        Returns:
            状态价值
        """
        return self.network(state)

class PPOAgent:
    """PPO代理，实现PPO算法"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-4, gamma=0.99, clip_epsilon=0.2, device="cuda" if torch.cuda.is_available() else "cpu"):
        """初始化PPO代理
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            clip_epsilon: PPO裁剪参数
            device: 计算设备
        """
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        
        # 初始化策略网络和价值网络
        self.policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(device)
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        
        # 经验缓冲区
        self.buffer = []
    
    def select_action(self, state):
        """选择动作
        
        Args:
            state: 状态向量
            
        Returns:
            选择的动作索引和动作概率
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            probs = self.policy_net(state_tensor)
            action = torch.multinomial(probs, 1).item()
            action_prob = probs[action].item()
        
        return action, action_prob
    
    def store_transition(self, state, action, action_prob, reward, next_state, done):
        """存储转换
        
        Args:
            state: 当前状态
            action: 执行的动作
            action_prob: 动作概率
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.buffer.append({
            'state': state,
            'action': action,
            'action_prob': action_prob,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def update(self, batch_size=32, epochs=10):
        """更新策略网络和价值网络
        
        Args:
            batch_size: 批量大小
            epochs: 更新轮数
        """
        if len(self.buffer) < batch_size:
            logger.info(f"缓冲区中的样本不足，跳过更新 ({len(self.buffer)}/{batch_size})")
            return
        
        # 从缓冲区中随机采样
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # 准备数据
        states = torch.FloatTensor([item['state'] for item in batch]).to(self.device)
        actions = torch.LongTensor([item['action'] for item in batch]).to(self.device)
        old_action_probs = torch.FloatTensor([item['action_prob'] for item in batch]).to(self.device)
        rewards = torch.FloatTensor([item['reward'] for item in batch]).to(self.device)
        next_states = torch.FloatTensor([item['next_state'] for item in batch]).to(self.device)
        dones = torch.FloatTensor([item['done'] for item in batch]).to(self.device)
        
        # 计算目标值和优势函数
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            
            # 计算目标值
            targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = targets - values
        
        # 多轮更新
        for _ in range(epochs):
            # 更新价值网络
            values = self.value_net(states).squeeze()
            value_loss = F.mse_loss(values, targets)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            # 更新策略网络
            probs = self.policy_net(states)
            action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # 计算比率
            ratios = action_probs / old_action_probs
            
            # 计算PPO目标
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
        # 清空缓冲区
        self.buffer = []
        
        logger.info(f"策略更新完成，价值损失: {value_loss.item():.4f}, 策略损失: {policy_loss.item():.4f}")
    
    def save(self, path):
        """保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
        }, path)
        logger.info(f"模型已保存到 {path}")
    
    def load(self, path):
        """加载模型
        
        Args:
            path: 加载路径
        """
        if not torch.cuda.is_available() and self.device == "cuda":
            # 如果没有CUDA但模型是在CUDA上训练的，加载到CPU
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(path)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        logger.info(f"模型已从 {path} 加载")
