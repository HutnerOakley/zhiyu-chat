# -*- coding: utf-8 -*-
"""
File Name: intent.py
Description: 意图识别模块
Author: Hunter
Time: 2026/5/12 22:05
"""

import json
import re
from typing import Any
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompts import INTENT_RECOGNIZE_PROMPT


@dataclass(frozen=True)
class IntentResult:
    """
    意图识别结果：
    - intents: 意图列表, 每个元素为一个意图名称
    - slots: slot值字典, 键为slot名称, 值为slot值
    - confidence: 置信度分数, 取值范围 0~1, 越大表示越信该意图
    """

    intents: list[str]
    slots: dict[str, Any]
    confidence: float


class IntentRecognizer:
    """
    意图识别器：通过大语言模型识别用户输入的意图，输出IntentResult对象
    """

    def __init__(self, llm):

        self._prompt = ChatPromptTemplate.from_messages(
            messages=[
                ("system", INTENT_RECOGNIZE_PROMPT),
                ("human", "{input}"),
            ]
        )
        self._llm = llm
        self._str_output_parser = StrOutputParser()


    def recognize(self, user_input: str) -> IntentResult:
        """
        识别意图
        :param user_input: 用户输入
        :return: IntentResult对象  意图识别结果
        """

        # 1.调用llm去进行意图识别，输出str格式的json字符串
        chain = self._prompt|self._llm|self._str_output_parser
        res = chain.invoke(input={"user_input":user_input})

        # 2.解析大模型输出
        data = self._parse_json(res)

        # 2.1 解析意图
        intents = data.get("intents", [])
        if not isinstance(intents, list):
            intents = data.get("intent", [])
            intents = [intents] if isinstance(intents, str) else []

        # 2.2 解析槽位
        slots = data.get("slots") if isinstance(data.get("slots") , dict) else {}

        # 2.3 解析confidence置信度
        try:
            confidence = float(data.get("confidence", 0.0))
        except ValueError:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return IntentResult(intents=intents, slots=slots, confidence=confidence)

    def _parse_json(self, text: str) -> dict[str, Any]:
        """
        解析为字典类型
        :param text: str
        :return:字典
        """

        # 如果模型输出为空，那么返回默认值
        if not text and not text.strip():
            return {"intents": ["general"], "slots": {}, "confidence": 0.0}

        text_str = text.strip()
        try:
            return json.loads(text_str)
        except json.JSONDecodeError:
            pass

        # 如果解析失败，尝试从模型输出中提取json字符串
        find_text = re.search(r"{{.*?}}", text_str,re.DOTALL)
        if find_text:
            try:
                return json.loads(find_text.group(0))
            except json.JSONDecodeError:
                pass

        # 如果未找到json字符串，那么返回默认值
        return {"intents": ["general"], "slots": {}, "confidence": 0.0}

