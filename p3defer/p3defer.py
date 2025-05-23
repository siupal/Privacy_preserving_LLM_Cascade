"""
P³Defer: 隐私保护LLM级联框架
基于论文《Privacy-preserved LLM Cascade via CoT-enhanced Policy Learning》的实现
"""

import os
import torch
import numpy as np
import json
import logging
import time
from transformers import AutoTokenizer
from rouge_score import rouge_scorer

# 导入自定义模块
from .models.ollama_client import OllamaClient
from .models.private_memory import PrivateMemory
from .models.policy_network import PPOAgent

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

class P3Defer:
    """P³Defer框架主类"""
    
    def __init__(self, 
                 local_model_name="gemma:2b",
                 server_model_name=None,
                 privacy_memory=None,
                 state_dim=96,
                 action_dim=3,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """初始化P³Defer框架
        
        Args:
            local_model_name: 本地模型名称
            server_model_name: 服务器模型名称，如果为None则暂时留空
            privacy_memory: 私有内存实例，如果为None则创建新实例
            state_dim: 状态向量维度
            action_dim: 动作空间维度
            device: 计算设备
        """
        self.device = device
        logger.info(f"使用设备: {device}")
        
        # 初始化本地模型
        self.local_model = OllamaClient(model_name=local_model_name)
        logger.info(f"初始化本地模型: {local_model_name}")
        
        # 初始化服务器模型（如果提供）
        self.server_model = None
        self.server_model_name = server_model_name
        
        if server_model_name and server_model_name.lower() != 'none':
            try:
                self.server_model = OllamaClient(model_name=server_model_name)
                logger.info(f"初始化服务器模型: {server_model_name}")
            except Exception as e:
                logger.error(f"初始化服务器模型失败: {e}")
                self.server_model = None
        
        # 初始化tokenizer（用于状态编码）
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
            logger.info("成功加载tokenizer")
        except Exception as e:
            logger.warning(f"加载tokenizer失败: {e}，使用备用方法")
            # 如果无法加载tokenizer，使用简单的分词方法
            self.tokenizer = None
        
        # 初始化私有内存
        if privacy_memory is None:
            self.privacy_memory = PrivateMemory()
        else:
            self.privacy_memory = privacy_memory
        
        # 初始化PPO代理
        self.ppo_agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
        
        # 初始化Rouge评分器（用于计算奖励）
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
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
        privacy_embedding = [1.0, 0.0] if privacy_detected else [0.0, 1.0]
        quality_embedding = [1.0, 0.0] if quality_good else [0.0, 1.0]
        
        # 如果有tokenizer，使用它来获取更丰富的表示
        if self.tokenizer:
            # 获取查询和输出的嵌入
            query_tokens = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128).input_ids
            output_tokens = self.tokenizer(local_output, return_tensors="pt", truncation=True, max_length=128).input_ids
            
            # 简单平均嵌入
            query_embedding = torch.mean(torch.nn.functional.one_hot(query_tokens, num_classes=32000).float(), dim=1).squeeze().tolist()
            output_embedding = torch.mean(torch.nn.functional.one_hot(output_tokens, num_classes=32000).float(), dim=1).squeeze().tolist()
            
            # 降维（取前面的一部分特征）
            query_embedding = query_embedding[:382]
            output_embedding = output_embedding[:382]
        else:
            # 如果没有tokenizer，使用简单的one-hot编码
            # 将查询和输出转换为字符级one-hot编码
            chars = set("abcdefghijklmnopqrstuvwxyz0123456789 ,.?!，。？！")
            char_to_idx = {c: i for i, c in enumerate(chars)}
            
            # 对查询进行编码
            query_chars = [char_to_idx.get(c.lower(), len(chars)) for c in query[:100]]
            query_embedding = np.zeros(len(chars) + 1)
            for c in query_chars:
                query_embedding[c] += 1
            query_embedding = query_embedding / max(1, len(query_chars))
            
            # 对输出进行编码
            output_chars = [char_to_idx.get(c.lower(), len(chars)) for c in local_output[:100]]
            output_embedding = np.zeros(len(chars) + 1)
            for c in output_chars:
                output_embedding[c] += 1
            output_embedding = output_embedding / max(1, len(output_chars))
            
            # 转换为列表
            query_embedding = query_embedding.tolist()
            output_embedding = output_embedding.tolist()
        
        # 连接所有嵌入
        state = privacy_embedding + quality_embedding + query_embedding + output_embedding
        
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
        # 质量奖励
        if golden_output:
            # 如果有黄金标准输出，计算ROUGE分数
            scores = self.rouge_scorer.score(golden_output, output)
            quality_reward = scores['rougeL'].fmeasure
        else:
            # 否则使用启发式方法
            quality_reward = min(1.0, len(output) / 200)  # 长度奖励
            if "答案：" in output:
                quality_reward += 0.5  # 格式奖励
            quality_reward = min(1.0, quality_reward)  # 确保不超过1.0
        
        # 隐私奖励
        privacy_tokens = self.privacy_memory.detect_privacy(query)
        if privacy_tokens:
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
        prompt = instruction_prompt.format(question=query)
        local_output = self.local_model.generate(prompt)
        
        logger.info(f"本地模型输出: {local_output[:100]}...")
        
        # 编码状态
        state = self.encode_state(query, local_output)
        
        # 选择动作
        action, action_prob = self.ppo_agent.select_action(state)
        
        # 执行动作
        if action == 0:  # 接受本地输出
            logger.info("动作: 接受本地输出")
            final_output = local_output
        elif action == 1:  # 直接转发到服务器
            logger.info("动作: 直接转发到服务器")
            if self.server_model is None:
                logger.warning("服务器模型未配置或不可用，回退到本地输出")
                final_output = local_output
                # 在训练模式下，如果服务器模型不可用，则使用本地模型模拟服务器模型
                if train_mode and self.server_model_name and self.server_model_name.lower() != 'none':
                    logger.info("训练模式: 使用本地模型模拟服务器模型")
            else:
                try:
                    # 使用服务器模型生成输出
                    final_output = self.server_model.generate(prompt)
                except Exception as e:
                    logger.error(f"服务器模型生成失败: {e}")
                    final_output = local_output
        else:  # 掩码后转发到服务器
            logger.info("动作: 掩码后转发到服务器")
            masked_query = self.privacy_memory.mask_privacy(query)
            logger.info(f"掩码后的查询: {masked_query}")
            
            if self.server_model is None:
                logger.warning("服务器模型未配置或不可用，回退到本地输出")
                final_output = local_output
                # 在训练模式下，如果服务器模型不可用，则使用本地模型模拟服务器模型
                if train_mode and self.server_model_name and self.server_model_name.lower() != 'none':
                    logger.info("训练模式: 使用本地模型模拟服务器模型")
                    # 对掩码后的查询再次使用本地模型
                    masked_prompt = instruction_prompt.format(question=masked_query)
                    final_output = self.local_model.generate(masked_prompt)
            else:
                try:
                    # 使用服务器模型生成输出
                    masked_prompt = instruction_prompt.format(question=masked_query)
                    final_output = self.server_model.generate(masked_prompt)
                except Exception as e:
                    logger.error(f"服务器模型生成失败: {e}")
                    final_output = local_output
        
        # 如果处于训练模式，计算奖励并存储经验
        if train_mode:
            reward = self.compute_reward(query, final_output, golden_output)
            
            # 编码下一个状态（简化版，使用相同的状态）
            next_state = state.copy()
            
            # 存储经验
            self.ppo_agent.store_transition(state, action, action_prob, reward, next_state, True)
            
            # 更新策略
            self.ppo_agent.update()
        
        return final_output
    
    def train(self, train_data, num_epochs=5, batch_size=32):
        """训练模型
        
        Args:
            train_data: 训练数据，每个元素应包含'question'和'answer'键
            num_epochs: 训练轮数
            batch_size: 批量大小
        """
        logger.info(f"开始训练，数据集大小: {len(train_data)}, 轮数: {num_epochs}")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # 打乱数据
            np.random.shuffle(train_data)
            
            total_reward = 0
            start_time = time.time()
            
            for i, item in enumerate(train_data):
                query = item['question']
                golden_output = item['answer']
                
                # 处理查询并更新策略
                output = self.process_query(query, train_mode=True, golden_output=golden_output)
                
                # 计算奖励（用于日志）
                reward = self.compute_reward(query, output, golden_output)
                total_reward += reward
                
                # 每处理10个查询，记录一次日志
                if (i + 1) % 10 == 0:
                    avg_reward = total_reward / (i + 1)
                    elapsed_time = time.time() - start_time
                    logger.info(f"已处理 {i+1}/{len(train_data)} 个查询，平均奖励: {avg_reward:.4f}，耗时: {elapsed_time:.2f}秒")
            
            # 每轮结束后保存模型
            self.save_model(f"p3defer_epoch_{epoch+1}.pt")
        
        logger.info("训练完成")
    
    def save_model(self, path):
        """保存模型
        
        Args:
            path: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # 保存PPO代理
        self.ppo_agent.save(path)
    
    def load_model(self, path):
        """加载模型
        
        Args:
            path: 加载路径
        """
        if os.path.exists(path):
            self.ppo_agent.load(path)
        else:
            logger.error(f"模型文件 {path} 不存在")
    
    @staticmethod
    def rouge_compute(predictions, references, rouge_types=None, use_aggregator=True):
        """计算ROUGE分数
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            rouge_types: ROUGE类型列表
            use_aggregator: 是否使用聚合器
            
        Returns:
            ROUGE分数
        """
        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL']
        
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=False)
        
        scores = []
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores.append(score)
        
        if use_aggregator:
            # 计算平均分数
            result = {}
            for key in rouge_types:
                result[key] = sum(score[key].fmeasure for score in scores) / len(scores)
        else:
            result = scores
        
        return result
