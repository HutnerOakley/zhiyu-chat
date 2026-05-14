# -*- coding: utf-8 -*-
"""
File Name: intent_with_structured_output.py
Description: 意图按识别转结构化输出
Author: Hunter
Time: 2026/5/13 19:55
"""

from typing import Any
from pydantic import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate
from config.prompts import INTENT_RECOGNIZE_WITH_STRUCTURED_OUTPUT_PROMPT


class IntentWithStructuredOutput(BaseModel):
    """
    意图识别结果结构化输出
    - intents: 意图列表, 每个元素为一个意图名称
    - slots: slot值字典, 键为slot名称, 值为slot值
    - confidence: 置信度分数, 取值范围 0~1, 越大表示越信该意图
    """
    intent: str = Field(..., description="意图名称")
    slots: dict[str, Any] = Field(..., description="slot值字典")
    confidence: float = Field(..., description="置信度分数")


class IntentRecognizer:
    """
    意图识别器：通过大语言模型识别用户输入的意图，输出IntentWithStructuredOutput对象
    """


    def __init__(self,llm):

        self._prompt = ChatPromptTemplate.from_messages(
            messages=[
                ("system", INTENT_RECOGNIZE_WITH_STRUCTURED_OUTPUT_PROMPT),
                ("human", "用户输入：{user_input}"),
            ]
        )
        self._llm = llm
        # 创建结构化输出大模型
        self.__structured_llm  =  self._llm.with_structured_output(IntentWithStructuredOutput)


    def recognize(self,user_input:str)->IntentWithStructuredOutput:
        """
        识别意图
        :param user_input: 用户输入
        :return: IntentWithStructuredOutput对象  意图识别结果
        """
        # 1 调用llm去进行意图识别
        chain = self._prompt|self.__structured_llm
        res= chain.invoke(
            input={"user_input":user_input}
        )

        if res is None:
            return IntentWithStructuredOutput(intent="general", slots={}, confidence=0.0)

        # 确保 confidence 在 0~1 范围内
        res.confidence = max(0.0,min(1.0,res.confidence))
        return res


