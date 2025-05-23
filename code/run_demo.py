"""
P³Defer框架演示脚本
"""

import argparse
import logging
import torch
import os
import json
import sys

# 添加当前目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入P³Defer框架
from code.p3defer import P3Defer
from code.models.private_memory import PrivateMemory

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='P³Defer框架演示')
    parser.add_argument('--mode', type=str, default='interactive', choices=['interactive', 'example'],
                        help='运行模式: interactive(交互式) 或 example(示例)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备: cuda 或 cpu')
    parser.add_argument('--local_model', type=str, default='gemma:2b',
                        help='本地模型名称')
    parser.add_argument('--server_model', type=str, default=None,
                        help='服务器模型名称 (可选)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='预训练模型路径 (可选)')
    parser.add_argument('--privacy_tokens', type=str, default=None,
                        help='隐私令牌文件路径 (可选)')
    return parser.parse_args()

def load_examples():
    """加载示例数据"""
    examples = [
        {
            "question": "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
            "description": "包含个人信息的简单数学问题"
        },
        {
            "question": "一个花园产出了237个土豆，60个黄瓜和黄瓜数量两倍的辣椒。这个花园总共产出了多少蔬菜？",
            "description": "不包含个人信息的复杂数学问题"
        },
        {
            "question": "李华每天读3本书，一周读几本书？",
            "description": "包含个人信息的简单数学问题"
        },
        {
            "question": "一个长方形的长是8厘米，宽是6厘米，求它的面积和周长。",
            "description": "不包含个人信息的几何问题"
        }
    ]
    return examples

def interactive_mode(p3defer):
    """交互式模式"""
    print("\n===== P³Defer交互式演示 =====")
    print("输入'exit'或'quit'退出\n")
    
    while True:
        query = input("\n请输入问题: ")
        if query.lower() in ['exit', 'quit']:
            print("退出演示")
            break
        
        print("处理中...")
        output = p3defer.process_query(query)
        print(f"\n输出: {output}")

def example_mode(p3defer):
    """示例模式"""
    examples = load_examples()
    
    print("\n===== P³Defer示例演示 =====")
    
    for i, example in enumerate(examples):
        print(f"\n示例 {i+1}: {example['description']}")
        print(f"问题: {example['question']}")
        
        print("处理中...")
        output = p3defer.process_query(example['question'])
        
        print(f"输出: {output}")
        print("-" * 50)

def main():
    args = parse_args()
    
    print(f"初始化P³Defer框架 (设备: {args.device})...")
    
    # 创建私有内存实例
    privacy_memory = PrivateMemory(args.privacy_tokens)
    
    # 创建P³Defer实例
    p3defer = P3Defer(
        local_model_name=args.local_model,
        server_model_name=args.server_model,
        privacy_memory=privacy_memory,
        device=args.device
    )
    
    # 如果提供了模型路径，加载预训练模型
    if args.model_path:
        print(f"加载预训练模型: {args.model_path}")
        p3defer.load_model(args.model_path)
    
    # 根据模式运行演示
    if args.mode == 'interactive':
        interactive_mode(p3defer)
    else:
        example_mode(p3defer)
    
    print("\n演示完成！")

if __name__ == "__main__":
    main()
