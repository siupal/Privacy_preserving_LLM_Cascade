"""
P³Defer框架训练脚本
"""

import argparse
import logging
import torch
import os
import json
import sys
import numpy as np
from tqdm import tqdm

# 添加当前目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入P³Defer框架
from code.p3defer import P3Defer
from code.models.private_memory import PrivateMemory

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='P³Defer框架训练')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备: cuda 或 cpu')
    parser.add_argument('--local_model', type=str, default='gemma:2b',
                        help='本地模型名称')
    parser.add_argument('--server_model', type=str, default=None,
                        help='服务器模型名称 (可选)')
    parser.add_argument('--train_data', type=str, required=True,
                        help='训练数据文件路径 (JSON格式)')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='模型输出目录')
    parser.add_argument('--privacy_tokens', type=str, default=None,
                        help='隐私令牌文件路径 (可选)')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()

def load_train_data(file_path):
    """加载训练数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        训练数据列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据格式
        if not isinstance(data, list):
            logger.error(f"数据格式错误: 应为列表，实际为 {type(data)}")
            return []
        
        # 检查每个数据项是否包含必要的键
        valid_data = []
        for item in data:
            if isinstance(item, dict) and 'question' in item and 'answer' in item:
                valid_data.append(item)
            else:
                logger.warning(f"跳过无效数据项: {item}")
        
        logger.info(f"成功加载 {len(valid_data)}/{len(data)} 条训练数据")
        return valid_data
    
    except Exception as e:
        logger.error(f"加载训练数据时出错: {e}")
        return []

def create_mock_data(num_samples=10):
    """创建模拟训练数据（当无法加载真实数据时使用）
    
    Args:
        num_samples: 样本数量
        
    Returns:
        模拟训练数据列表
    """
    logger.warning("使用模拟训练数据")
    
    # 包含个人信息的问题模板
    personal_templates = [
        "小明有{a}个苹果，小红有{b}个苹果，他们一共有多少个苹果？",
        "李华每天读{a}本书，一周读几本书？",
        "张三有{a}元钱，买了一本{b}元的书，还剩多少钱？",
        "王五比李四大{a}岁，李四今年{b}岁，王五今年多少岁？"
    ]
    
    # 不包含个人信息的问题模板
    non_personal_templates = [
        "一个长方形的长是{a}厘米，宽是{b}厘米，求它的面积和周长。",
        "一个花园产出了{a}个土豆，{b}个黄瓜和黄瓜数量两倍的辣椒。这个花园总共产出了多少蔬菜？",
        "一辆汽车以每小时{a}公里的速度行驶了{b}小时，行驶了多少公里？",
        "一个数比{a}大{b}，这个数是多少？"
    ]
    
    mock_data = []
    
    for _ in range(num_samples):
        # 随机选择模板
        if np.random.random() < 0.5:
            template = np.random.choice(personal_templates)
            contains_personal = True
        else:
            template = np.random.choice(non_personal_templates)
            contains_personal = False
        
        # 生成随机数
        a = np.random.randint(1, 100)
        b = np.random.randint(1, 100)
        
        # 生成问题
        question = template.format(a=a, b=b)
        
        # 生成答案
        if "一共有多少个苹果" in question:
            answer = f"a. 包含个人信息：是\nb. 答案：{a}+{b}={a+b}个苹果"
        elif "一周读几本书" in question:
            answer = f"a. 包含个人信息：是\nb. 答案：{a}*7={a*7}本书"
        elif "还剩多少钱" in question:
            answer = f"a. 包含个人信息：是\nb. 答案：{a}-{b}={a-b}元"
        elif "王五今年多少岁" in question:
            answer = f"a. 包含个人信息：是\nb. 答案：{b}+{a}={b+a}岁"
        elif "面积和周长" in question:
            answer = f"a. 包含个人信息：否\nb. 答案：面积={a}*{b}={a*b}平方厘米，周长=2*({a}+{b})={2*(a+b)}厘米"
        elif "总共产出了多少蔬菜" in question:
            answer = f"a. 包含个人信息：否\nb. 答案：{a}+{b}+{b*2}={a+b+b*2}个蔬菜"
        elif "行驶了多少公里" in question:
            answer = f"a. 包含个人信息：否\nb. 答案：{a}*{b}={a*b}公里"
        elif "这个数是多少" in question:
            answer = f"a. 包含个人信息：否\nb. 答案：{a}+{b}={a+b}"
        else:
            answer = f"a. 包含个人信息：{'是' if contains_personal else '否'}\nb. 答案：计算结果"
        
        mock_data.append({
            "question": question,
            "answer": answer
        })
    
    return mock_data

def main():
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载训练数据
    train_data = load_train_data(args.train_data)
    
    # 如果无法加载训练数据，创建模拟数据
    if not train_data:
        train_data = create_mock_data(num_samples=50)
    
    # 创建私有内存实例
    privacy_memory = PrivateMemory(args.privacy_tokens)
    
    # 创建P³Defer实例
    p3defer = P3Defer(
        local_model_name=args.local_model,
        server_model_name=args.server_model,
        privacy_memory=privacy_memory,
        device=args.device
    )
    
    # 训练模型
    logger.info(f"开始训练，数据集大小: {len(train_data)}, 轮数: {args.num_epochs}")
    p3defer.train(train_data, num_epochs=args.num_epochs, batch_size=args.batch_size)
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "p3defer_final.pt")
    p3defer.save_model(final_model_path)
    logger.info(f"最终模型已保存到 {final_model_path}")
    
    # 保存私有内存
    privacy_tokens_path = os.path.join(args.output_dir, "privacy_tokens.json")
    privacy_memory.save_to_file(privacy_tokens_path)
    logger.info(f"隐私令牌已保存到 {privacy_tokens_path}")
    
    logger.info("训练完成")

if __name__ == "__main__":
    main()
