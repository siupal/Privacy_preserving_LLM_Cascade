"""
P³Defer框架单元测试
"""

import unittest
import sys
import os
import torch

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p3defer import P3Defer, PrivateMemory

class TestP3Defer(unittest.TestCase):
    """P³Defer框架测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.privacy_memory = PrivateMemory()
        self.p3defer = P3Defer(
            local_model_name="gemma:2b",
            privacy_memory=self.privacy_memory,
            device="cpu"  # 使用CPU进行测试
        )
    
    def test_privacy_detection(self):
        """测试隐私检测功能"""
        # 包含隐私信息的查询
        query_with_privacy = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
        privacy_tokens = self.privacy_memory.detect_privacy(query_with_privacy)
        self.assertTrue(len(privacy_tokens) > 0, "应该检测到隐私令牌")
        
        # 不包含隐私信息的查询
        query_without_privacy = "一个长方形的长是8厘米，宽是6厘米，求它的面积和周长。"
        privacy_tokens = self.privacy_memory.detect_privacy(query_without_privacy)
        self.assertEqual(len(privacy_tokens), 0, "不应该检测到隐私令牌")
    
    def test_state_encoding(self):
        """测试状态编码功能"""
        query = "测试查询"
        local_output = "测试输出"
        state = self.p3defer.encode_state(query, local_output)
        
        self.assertIsInstance(state, list, "状态应该是列表类型")
        self.assertTrue(len(state) > 0, "状态不应为空")
    
    def test_reward_computation(self):
        """测试奖励计算功能"""
        query = "测试查询"
        output = "测试输出"
        golden_output = "黄金标准输出"
        
        reward = self.p3defer.compute_reward(query, output, golden_output)
        self.assertIsInstance(reward, float, "奖励应该是浮点数类型")
        self.assertTrue(0 <= reward <= 1, "奖励应该在0到1之间")

if __name__ == "__main__":
    unittest.main()
