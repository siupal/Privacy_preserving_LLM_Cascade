"""
P³Defer: Privacy-preserved LLM Cascade via CoT-enhanced Policy Learning
基于论文《Privacy-preserved LLM Cascade via CoT-enhanced Policy Learning》的实现
"""

import os
import torch
import numpy as np
import json
import requests
import Levenshtein
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CoT增强的指令提示模板
instruction_prompt = r'''假设你是一名学生，正在解决数学问题。现在，你将被给予数学问题，请按照以下两个任务进行：
a. 检查问题是否包含个人信息（例如姓名等），只输出"是"或"否"；
b. 解决这个问题；

这里有一个例子：
问题：小明购买了一盒口香糖。他给了小张4颗，然后他给了小丽的数量是他给小张的两倍，然后他给了小刚的数量是他给小丽的四倍减去5颗。如果小明还剩下6颗口香糖，小明最初购买了多少颗口香糖？
输出：
让我们一步步思考：
对于a，问题中包含个人姓名，所以答案是"是"。
对于b，小明给小丽的口香糖数量是他给小张的两倍，总共是4*2=8颗口香糖，小明给小刚的数量是他给小丽的四倍减去5，总共是(8*4)-5=27颗口香糖。如果小明还剩下6颗口香糖，他最初购买的口香糖数量是4+8+27+6=45颗口香糖。
a. 包含个人信息：是
b. 答案：45

现在，
问题：{question}
输出：
让我们一步步思考：
'''

class PrivateMemory:
    """私有内存类，用于存储和检测隐私令牌"""
    
    def __init__(self, privacy_tokens_file=None):
        """初始化私有内存
        
        Args:
            privacy_tokens_file: 包含隐私令牌的文件路径，如果为None则使用默认令牌
        """
        self.privacy_tokens = set()
        
        # 如果提供了文件，从文件加载隐私令牌
        if privacy_tokens_file and os.path.exists(privacy_tokens_file):
            with open(privacy_tokens_file, 'r', encoding='utf-8') as f:
                self.privacy_tokens = set(json.load(f))
        else:
            # 默认的隐私令牌列表
            self.privacy_tokens = {
                "姓名", "名字", "小明", "小红", "小张", "小李", "小王", "张三", "李四", "王五", 
                "电话", "手机", "地址", "邮箱", "身份证", "账号", "密码", "银行卡",
                "病历", "病情", "症状", "诊断", "医院", "医生"
            }
        
        self.memory_dict = {}  # 用于存储检测到的隐私令牌
    
    def detect_privacy(self, text, threshold=0.8):
        """检测文本中的隐私令牌
        
        Args:
            text: 要检查的文本
            threshold: Levenshtein距离的阈值，用于模糊匹配
            
        Returns:
            检测到的隐私令牌列表
        """
        detected_tokens = []
        
        # 将文本分割成词
        words = text.split()
        
        for word in words:
            # 精确匹配
            if word in self.privacy_tokens:
                detected_tokens.append(word)
                self.memory_dict[len(self.memory_dict)] = word
                continue
            
            # 模糊匹配
            for token in self.privacy_tokens:
                # 计算Levenshtein距离的相似度
                if len(token) > 0 and len(word) > 0:
                    similarity = 1 - Levenshtein.distance(word, token) / max(len(word), len(token))
                    if similarity >= threshold:
                        detected_tokens.append(word)
                        self.memory_dict[len(self.memory_dict)] = word
                        break
        
        return detected_tokens
    
    def mask_privacy(self, text, mask_token="[MASK]"):
        """掩码文本中的隐私令牌
        
        Args:
            text: 要掩码的文本
            mask_token: 用于替换隐私令牌的掩码标记
            
        Returns:
            掩码后的文本
        """
        detected_tokens = self.detect_privacy(text)
        masked_text = text
        
        for token in detected_tokens:
            masked_text = masked_text.replace(token, mask_token)
        
        return masked_text

class QueryDataset(Dataset):
    """查询数据集类"""
    
    def __init__(self, data, tokenizer, max_length=512):
        """初始化数据集
        
        Args:
            data: 数据列表，每个元素应包含'question'和'answer'键
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text = instruction_prompt.format(question=self.data[idx]['question'])
        output_text = self.data[idx]['answer']
        
        encoded = self.tokenizer(
            input_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        
        labels = self.tokenizer(
            output_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        
        encoded["labels"] = labels["input_ids"]
        return {key: val.squeeze(0) for key, val in encoded.items()}

class OllamaClient:
    """Ollama API客户端，用于与本地Ollama模型交互"""
    
    def __init__(self, model_name="gemma:2b", base_url="http://localhost:11434"):
        """初始化Ollama客户端
        
        Args:
            model_name: Ollama模型名称
            base_url: Ollama API的基础URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        
        # 测试连接
        try:
            response = requests.get(f"{base_url}/api/version")
            if response.status_code == 200:
                logger.info(f"成功连接到Ollama服务，版本: {response.json().get('version')}")
            else:
                logger.warning("无法获取Ollama版本信息")
        except Exception as e:
            logger.error(f"连接Ollama服务失败: {e}")
    
    def generate(self, prompt, max_tokens=2048, temperature=0.7):
        """生成文本
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成令牌数
            temperature: 采样温度
            
        Returns:
            生成的文本
        """
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            response = requests.post(self.generate_url, json=data)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama API错误: {response.status_code}, {response.text}")
                return ""
        except Exception as e:
            logger.error(f"调用Ollama API时出错: {e}")
            return ""

class PolicyNetwork(torch.nn.Module):
    """策略网络，用于决策延迟动作"""
    
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=3):
        """初始化策略网络
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（动作数）
        """
        super(PolicyNetwork, self).__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        """前向传播
        
        Args:
            state: 状态向量
            
        Returns:
            动作概率分布
        """
        return self.network(state)

class ValueNetwork(torch.nn.Module):
    """价值网络，用于评估状态价值"""
    
    def __init__(self, input_dim=768, hidden_dim=256):
        """初始化价值网络
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        super(ValueNetwork, self).__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """前向传播
        
        Args:
            state: 状态向量
            
        Returns:
            状态价值
        """
        return self.network(state)

class P3Defer:
    """P³Defer框架主类"""
    
    def __init__(self, 
                 local_model=None,
                 server_model=None,
                 privacy_memory=None,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """初始化P³Defer框架
        
        Args:
            local_model: 本地模型，如果为None则使用Ollama客户端
            server_model: 服务器模型，如果为None则暂时留空
            privacy_memory: 私有内存实例，如果为None则创建新实例
            device: 计算设备
        """
        self.device = device
        logger.info(f"使用设备: {device}")
        
        # 初始化本地模型
        if local_model is None:
            logger.info("使用Ollama客户端作为本地模型")
            self.local_model = OllamaClient(model_name="gemma:2b")
            # 为了兼容性，我们仍然需要一个tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        else:
            logger.info("使用提供的本地模型")
            self.local_model = local_model
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        
        # 初始化服务器模型（暂时留空）
        self.server_model = server_model
        
        # 初始化私有内存
        if privacy_memory is None:
            logger.info("创建新的私有内存实例")
            self.privacy_memory = PrivateMemory()
        else:
            logger.info("使用提供的私有内存实例")
            self.privacy_memory = privacy_memory
        
        # 初始化策略网络和价值网络
        self.policy_net = PolicyNetwork().to(device)
        self.value_net = ValueNetwork().to(device)
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-4)
        
        # 经验缓冲区
        self.experience_buffer = []
    
    def encode_state(self, query, local_output):
        """编码状态
        
        Args:
            query: 用户查询
            local_output: 本地模型输出
            
        Returns:
            状态向量
        """
        # 检测隐私
        privacy_detected = len(self.privacy_memory.detect_privacy(query)) > 0
        
        # 评估质量（简化版，实际应根据具体任务调整）
        # 这里我们简单地检查输出长度和是否包含"答案："作为质量指标
        quality_good = len(local_output) > 50 and "答案：" in local_output
        
        # 创建状态向量
        # [1,0]表示包含隐私，[0,1]表示不包含隐私
        # [1,0]表示质量好，[0,1]表示质量差
        privacy_embedding = torch.tensor([1.0, 0.0] if privacy_detected else [0.0, 1.0], device=self.device)
        quality_embedding = torch.tensor([1.0, 0.0] if quality_good else [0.0, 1.0], device=self.device)
        
        # 将嵌入连接成状态向量
        # 为了扩展状态表示，我们使用tokenizer获取查询和输出的嵌入
        query_tokens = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128).input_ids.to(self.device)
        output_tokens = self.tokenizer(local_output, return_tensors="pt", truncation=True, max_length=128).input_ids.to(self.device)
        
        # 简单平均嵌入
        query_embedding = torch.mean(torch.nn.functional.one_hot(query_tokens, num_classes=32000).float(), dim=1).squeeze()
        output_embedding = torch.mean(torch.nn.functional.one_hot(output_tokens, num_classes=32000).float(), dim=1).squeeze()
        
        # 降维（取前面的一部分特征）
        query_embedding = query_embedding[:382]
        output_embedding = output_embedding[:382]
        
        # 连接所有嵌入
        state = torch.cat([privacy_embedding, quality_embedding, query_embedding, output_embedding], dim=0)
        
        return state
    
    def compute_reward(self, query, output, golden_output=None, privacy_weight=0.5):
        """计算奖励
        
        Args:
            query: 用户查询
            output: 最终输出
            golden_output: 黄金标准输出（如果有）
            privacy_weight: 隐私奖励的权重
            
        Returns:
            奖励值
        """
        # 质量奖励（简化版）
        if golden_output is not None:
            # 如果有黄金标准输出，计算相似度
            quality_reward = 1.0 - Levenshtein.distance(output, golden_output) / max(len(output), len(golden_output))
        else:
            # 否则使用启发式方法
            quality_reward = min(1.0, len(output) / 200)  # 长度奖励
            if "答案：" in output:
                quality_reward += 0.5  # 格式奖励
        
        # 隐私奖励
        privacy_tokens = self.privacy_memory.detect_privacy(query)
        if len(privacy_tokens) > 0:
            # 如果查询包含隐私令牌，检查它们是否泄露到输出中
            leaked_tokens = 0
            for token in privacy_tokens:
                if token in output:
                    leaked_tokens += 1
            
            privacy_reward = 1.0 - (leaked_tokens / len(privacy_tokens))
        else:
            # 如果查询不包含隐私令牌，隐私奖励为1
            privacy_reward = 1.0
        
        # 总奖励
        total_reward = (1 - privacy_weight) * quality_reward + privacy_weight * privacy_reward
        
        return total_reward
    
    def select_action(self, state):
        """选择动作
        
        Args:
            state: 状态向量
            
        Returns:
            选择的动作索引
        """
        with torch.no_grad():
            probs = self.policy_net(state)
            action = torch.multinomial(probs, 1).item()
        
        return action
    
    def update_policy(self, batch_size=32, gamma=0.99, clip_epsilon=0.2):
        """更新策略网络和价值网络
        
        Args:
            batch_size: 批量大小
            gamma: 折扣因子
            clip_epsilon: PPO裁剪参数
        """
        if len(self.experience_buffer) < batch_size:
            logger.info(f"经验缓冲区中的样本不足，跳过更新 ({len(self.experience_buffer)}/{batch_size})")
            return
        
        # 从经验缓冲区中随机采样
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        states = torch.stack([item['state'] for item in batch])
        actions = torch.tensor([item['action'] for item in batch], device=self.device)
        rewards = torch.tensor([item['reward'] for item in batch], device=self.device)
        next_states = torch.stack([item['next_state'] for item in batch])
        dones = torch.tensor([item['done'] for item in batch], device=self.device)
        
        # 计算优势函数
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            
            # 计算目标值
            targets = rewards + gamma * next_values * (1 - dones)
            advantages = targets - values
        
        # 更新价值网络
        for _ in range(5):  # 多次更新价值网络
            values = self.value_net(states).squeeze()
            value_loss = torch.nn.functional.mse_loss(values, targets)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # 更新策略网络
        for _ in range(5):  # 多次更新策略网络
            # 计算当前策略的动作概率
            probs = self.policy_net(states)
            action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # 计算旧策略的动作概率
            with torch.no_grad():
                old_probs = self.policy_net(states)
                old_action_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # 计算比率
            ratio = action_probs / (old_action_probs + 1e-8)
            
            # 计算PPO目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
        logger.info(f"策略更新完成，价值损失: {value_loss.item():.4f}, 策略损失: {policy_loss.item():.4f}")
    
    def process_query(self, query, train_mode=False, golden_output=None):
        """处理用户查询
        
        Args:
            query: 用户查询
            train_mode: 是否处于训练模式
            golden_output: 黄金标准输出（仅在训练模式下使用）
            
        Returns:
            最终输出
        """
        # 生成本地模型输出
        if isinstance(self.local_model, OllamaClient):
            prompt = instruction_prompt.format(question=query)
            local_output = self.local_model.generate(prompt)
        else:
            # 如果是Hugging Face模型
            inputs = self.tokenizer(instruction_prompt.format(question=query), return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.local_model.generate(**inputs, max_length=512)
            local_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"本地模型输出: {local_output[:100]}...")
        
        # 编码状态
        state = self.encode_state(query, local_output)
        
        # 选择动作
        action = self.select_action(state)
        
        # 执行动作
        if action == 0:  # 接受本地输出
            logger.info("动作: 接受本地输出")
            final_output = local_output
        elif action == 1:  # 直接转发到服务器
            logger.info("动作: 直接转发到服务器")
            if self.server_model is None:
                logger.warning("服务器模型未配置，回退到本地输出")
                final_output = local_output
            else:
                # 使用服务器模型生成输出
                inputs = self.tokenizer(instruction_prompt.format(question=query), return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.server_model.generate(**inputs, max_length=512)
                final_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:  # 掩码后转发到服务器
            logger.info("动作: 掩码后转发到服务器")
            masked_query = self.privacy_memory.mask_privacy(query)
            logger.info(f"掩码后的查询: {masked_query}")
            
            if self.server_model is None:
                logger.warning("服务器模型未配置，回退到本地输出")
                final_output = local_output
            else:
                # 使用服务器模型生成输出
                inputs = self.tokenizer(instruction_prompt.format(question=masked_query), return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.server_model.generate(**inputs, max_length=512)
                final_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 如果处于训练模式，计算奖励并存储经验
        if train_mode:
            reward = self.compute_reward(query, final_output, golden_output)
            
            # 编码下一个状态（简化版，使用相同的状态）
            next_state = state.clone()
            
            # 存储经验
            self.experience_buffer.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': True  # 每个查询都是一个完整的轨迹
            })
            
            # 更新策略
            if len(self.experience_buffer) >= 32:
                self.update_policy()
        
        return final_output
    
    def train(self, train_data, num_epochs=5):
        """训练模型
        
        Args:
            train_data: 训练数据，每个元素应包含'question'和'answer'键
            num_epochs: 训练轮数
        """
        logger.info(f"开始训练，数据集大小: {len(train_data)}, 轮数: {num_epochs}")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            for i, item in enumerate(tqdm(train_data)):
                query = item['question']
                golden_output = item['answer']
                
                # 处理查询并更新策略
                self.process_query(query, train_mode=True, golden_output=golden_output)
                
                # 每处理100个查询，记录一次日志
                if (i + 1) % 100 == 0:
                    logger.info(f"已处理 {i+1}/{len(train_data)} 个查询")
            
            # 每轮结束后保存模型
            self.save_model(f"p3defer_epoch_{epoch+1}.pt")
        
        logger.info("训练完成")
    
    def save_model(self, path):
        """保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
        }, path)
        logger.info(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """加载模型
        
        Args:
            path: 加载路径
        """
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.value_net.load_state_dict(checkpoint['value_net'])
            logger.info(f"模型已从 {path} 加载")
        else:
            logger.error(f"模型文件 {path} 不存在")

# 示例用法
if __name__ == "__main__":
    # 创建P³Defer实例
    p3defer = P3Defer()
    
    # 示例查询
    query = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
    
    # 处理查询
    output = p3defer.process_query(query)
    
    print(f"查询: {query}")
    print(f"输出: {output}")
