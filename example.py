"""
P³Defer框架使用示例
"""

import json
import torch
from p3defer import P3Defer, PrivateMemory

# 示例数据
sample_data = [
    {
        "question": "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
        "answer": "a. 包含个人信息：是\nb. 答案：5+3=8个苹果"
    },
    {
        "question": "一个花园产出了237个土豆，60个黄瓜和黄瓜数量两倍的辣椒。这个花园总共产出了多少蔬菜？",
        "answer": "a. 包含个人信息：否\nb. 答案：237+60+(60*2)=417个蔬菜"
    },
    {
        "question": "李华每天读3本书，一周读几本书？",
        "answer": "a. 包含个人信息：是\nb. 答案：3*7=21本书"
    }
]

def main():
    print("初始化P³Defer框架...")
    
    # 创建私有内存实例
    privacy_memory = PrivateMemory()
    
    # 创建P³Defer实例
    p3defer = P3Defer(privacy_memory=privacy_memory)
    
    # 演示处理查询
    print("\n===== 演示处理查询 =====")
    for i, item in enumerate(sample_data):
        print(f"\n处理查询 {i+1}:")
        print(f"问题: {item['question']}")
        
        # 处理查询
        output = p3defer.process_query(item['question'])
        
        print(f"输出: {output}")
        print(f"期望输出: {item['answer']}")
        print("-" * 50)
    
    # 演示训练过程
    print("\n===== 演示训练过程 =====")
    print("开始小规模训练...")
    p3defer.train(sample_data, num_epochs=2)
    
    # 训练后再次处理查询
    print("\n===== 训练后再次处理查询 =====")
    for i, item in enumerate(sample_data):
        print(f"\n处理查询 {i+1}:")
        print(f"问题: {item['question']}")
        
        # 处理查询
        output = p3defer.process_query(item['question'])
        
        print(f"输出: {output}")
        print(f"期望输出: {item['answer']}")
        print("-" * 50)
    
    print("\n演示完成！")

if __name__ == "__main__":
    main()
