# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def set_learning_rate(optimizer, lr):   # 设置优化器的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ResidualBlock(nn.Module):
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class Net(nn.Module):
    
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        
        self.board_width = board_width
        self.board_height = board_height
        
        self.input_conv = nn.Conv2d(8, 64, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(64)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(10)
        ])
        
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_width * board_height, 
                                   board_width * board_height)
        
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_width * board_height, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, state_input):
        x = F.relu(self.input_bn(self.input_conv(state_input)))
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_width * self.board_height)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 1 * self.board_width * self.board_height)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


class PolicyValueNet:
    
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4   # L2正则化系数 用于防止过拟合
        
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.policy_value_net = Net(board_width, board_height).to(self.device)
        else:
            self.device = torch.device('cpu')
            self.policy_value_net = Net(board_width, board_height)
        
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(),
            weight_decay=self.l2_const    # L2正则化
        )
        
        if model_file:
            self.load_model(model_file)
    
    def policy_value(self, state_batch):   # 输入一批状态 返回每个状态的动作概率和状态价值 用于训练或批量评估
        state_batch = np.array(state_batch, dtype=np.float32)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        
        self.policy_value_net.eval()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = torch.exp(log_act_probs).cpu().numpy()
            value = value.cpu().numpy()
        
        return act_probs, value
    
    def policy_value_fn(self, board):   # 输入单个状态 返回动作概率和状态价值 用于MCTS搜索
        legal_positions = board.availables
        
        current_state = np.ascontiguousarray(
            board.current_state().reshape(-1, 8, self.board_width, self.board_height)
        )
        
        state_tensor = torch.FloatTensor(current_state).to(self.device)
        
        self.policy_value_net.eval()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_tensor)
            act_probs = torch.exp(log_act_probs).cpu().numpy().flatten()
            value = value.cpu().numpy()[0][0]
        
        act_probs_dict = [(pos, act_probs[pos]) for pos in legal_positions]
        
        return act_probs_dict, value
    
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        self.policy_value_net.train()
        
        set_learning_rate(self.optimizer, lr)   # 设置当前学习率
        
        state_batch = np.array(state_batch, dtype=np.float32)   # 转为numpy数组
        mcts_probs = np.array(mcts_probs, dtype=np.float32)
        winner_batch = np.array(winner_batch, dtype=np.float32)
    
        state_batch = torch.FloatTensor(state_batch).to(self.device)   # 转为张量并放到对应设备上
        mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
        winner_batch = torch.FloatTensor(winner_batch).to(self.device)
        
        self.optimizer.zero_grad()   # 清空梯度避免梯度累积
        log_act_probs, value = self.policy_value_net(state_batch)   # 前向传播
        
        value_loss = F.mse_loss(value.view(-1), winner_batch)   # 计算均方误差损失
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))   # 计算交叉熵损失
        loss = value_loss + policy_loss
        
        loss.backward()   # 反向传播计算梯度
        self.optimizer.step()   # 更新网络参数
        
        act_probs = torch.exp(log_act_probs)   # 对数概率还原为原始概率用于计算熵
        entropy = -torch.mean(torch.sum(act_probs * log_act_probs, dim=1))   # 策略熵 用于衡量策略的确定性
        
        return loss.item(), entropy.item()
    
    def get_policy_param(self):   # 获取网络的所有可学习参数
        return self.policy_value_net.state_dict()
    
    def save_model(self, model_file):   # 保存模型参数到指定路径
        torch.save(self.policy_value_net.state_dict(), model_file)
        print(f"Model saved to {model_file}")
    
    def load_model(self, model_file):   # 从指定路径加载模型参数
        if self.use_gpu and torch.cuda.is_available():
            net_params = torch.load(model_file, map_location=torch.device('cpu'))
            self.policy_value_net.load_state_dict(net_params)
            self.policy_value_net.to(self.device)
        else:
            net_params = torch.load(model_file, map_location=torch.device('cpu'))
            self.policy_value_net.load_state_dict(net_params)
        
        self.policy_value_net.eval()
        print(f"Model loaded from {model_file}")