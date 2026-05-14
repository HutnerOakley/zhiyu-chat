# -*- coding: utf-8 -*-
"""
File Name: 测试chat_service.py
Description: 
Author: Hunter
Time: 2026/5/15 1:25
"""

import uuid
import os
from core.protocol import ChatRequest
from services.chat_service import ChatService


if __name__ == '__main__':

    print("开始测试ChatService...")
    print(f"当前工作目录: {os.getcwd()}")

    # 创建测试请求
    request = ChatRequest(
        message_id=str(uuid.uuid4()),
        session_id="test_session",
        user_id="test_user",
        user_input="我的快递到哪了？我的订单号是7856757536124",
        trace_id=str(uuid.uuid4())
    )
    print(f"测试请求: {request}")

    try:
        chat_service = ChatService()
        print("ChatService初始化成功")
        response = chat_service.handle(request)
        print(f"响应: {response}")
        print("测试通过！")

    except Exception as e:
        print(f"创建ChatService实例时出错: {e}")
