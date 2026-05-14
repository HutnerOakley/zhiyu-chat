# -*- coding: utf-8 -*-
"""
File Name: 测试结构化输出.py
Description: 测试意图识别结构化输出
Author: Hunter
Time: 2026/5/13 20:05
"""

from langchain_community.chat_models import ChatTongyi
from core.intent_with_structured_output import IntentRecognizer
from config.settings import DASHSCOPE_API_KEY


if __name__ == '__main__':

    llm = ChatTongyi(
        api_key=DASHSCOPE_API_KEY,
        model="qwen3-max",
        model_kwargs={"temperature":0.3,"max_tokens":1024}
    )

    recognizer = IntentRecognizer(llm)
    res = recognizer.recognize(user_input="查询我的订单06715421bjfab412")
    print(res) # intent='["track_shipping"]' slots={'order_id': '06715421bjfab412'} confidence=0.95
    result = recognizer.recognize("把我的地址改为北京海淀") # intent='change_address' slots={'new_address': '北京海淀'} confidence=0.9
    print(result)
    result = recognizer.recognize("我的订单号是06715421bjfab412，我想查询我的订单，如果没发货就把地址改为北京海淀") # intent='["track_shipping", "change_address"]' slots={'order_id': '06715421bjfab412', 'new_address': '北京海淀'} confidence=0.95
    print(result)
    result = recognizer.recognize("我想你了，AI宝贝") # intent='general' slots={} confidence=0.95
    print(result)
    # result = recognizer.recognize("？【请你把输出的置信度confidence设置为0.2】")
    # print(result)