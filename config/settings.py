# -*- coding: utf-8 -*-
"""
File Name: settings.py
Description: 配置
Author: Hunter
Time: 2026/5/12 22:13
"""


import os

# 千问模型密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY","")

# 模型名称：就是你要使用阿里云百炼平台里面的哪个模型
TONGYI_MODEL = "qwen-turbo"

# 模型温度参数
"""
如果temperature越大，模型的输出越随机；
如果temperature越小，模型的输出越确定。
"""
TONGYI_TEMPERATURE = 0.3

# 模型最大生成长度 :用于限制模型的输出长度，防止模型输出过长。
TONGYI_MAX_TOKEN = 1024

# 记忆近期保留的对话轮数
RECENT_WINDOW = 6
# 每6(2 * 6 = 12)轮对话生成一次摘要
SUMMARY_BATCH_SIZE = 12

# 模型置信度阈值
CONFIDENCE_THRESHOLD = 0.3