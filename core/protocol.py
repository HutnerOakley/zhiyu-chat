# -*- coding: utf-8 -*-
"""
File Name: protocol.py
Description:  统一请求跟响应
Author: Hunter
Time: 2026/5/12 23:27
"""

from dataclasses import dataclass



@dataclass(frozen=True)
class ChatRequest:
    """
    聊天请求
     - user_id: 用户ID
    - session_id: 会话ID (user_id + session_id就可以保证会话的唯一性)
    - message_id: 消息ID (可以用于做消息幂等性处理，避免重复发送相同的消息)
    - trace_id: 跟踪ID (用于日志记录和调试)
    - user_input: 用户输入的文本
    """
    message_id: str
    session_id: str
    user_id: str
    user_input: str
    trace_id: str = ""


@dataclass(frozen=True)
class ChatResponse:
    """
    聊天响应
      - user_id: 用户ID
    - session_id: 会话ID
    - message_id: 消息ID
    - trace_id: 跟踪ID
    - response: 大模型的响应文本
    """
    message_id: str
    session_id: str
    user_id: str
    trace_id: str
    response: str
