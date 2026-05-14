# -*- coding: utf-8 -*-
"""
File Name: memory.py
Description: 记忆处理
Author: Hunter
Time: 2026/5/12 22:04
"""
from typing import Dict,List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from config.settings import RECENT_WINDOW,SUMMARY_BATCH_SIZE
from config.prompts import SUMMARY_GENERATION_PROMPT


# 存储每个用户的对话历史: key是user_id:session, value是BaseChatMessageHistory对象,会话历史记录
_session_store:Dict[str,BaseChatMessageHistory] = {}
# 存储每个用户的摘要: key是user_id:session_id, value是摘要
_summary_store:Dict[str,str] = {}
# 存储每个用的关键事实(slots): key是user_id:session, value是slots
_fact_store:Dict[str,Dict[str,str]] = {}
# 存储每个用户的未摘要消息数量：key是user_id:session, value是未摘要消息数量
_unsummarized_message_count:Dict[str,int] = {}


class MemoryHandler:
    """
    记忆处理类
    """


    def __init__(self,user_id:str,session_id:str):

        self._user_id = user_id
        self._session_id = session_id
        self._session_key = f"{self._user_id}:{self._session_id}"
        self.str_output_parser = StrOutputParser()
        # 初始化会话摘要,事实，未摘要消息数量
        _summary_store[self._session_key] = ""
        _fact_store[self._session_key] = {}
        _unsummarized_message_count[self._session_key] = 0


    def get_session_history(self)->BaseChatMessageHistory:
        """
        获取会话历史
        :return: BaseChatMessageHistory对象
        """

        if self._session_key not in _session_store:
            # 初始化会话历史记录
            _session_store[self._session_key] = ChatMessageHistory()
        return _session_store[self._session_key]


    def _generate_incremental_summary(self,old_summary:str,new_messages:List[BaseMessage],llm=None)->str:
        """
        增量摘要：将[旧摘要]和[新消息]合并后，调用大语言模型生成增量摘要
        :param old_summary: 旧摘要
        :param new_messages: 新的会话历史记录
        :param llm: 语言模型
        :return: 增量会话摘要
        """

        # 如果没有新的消息，直接返回旧的摘要
        if not new_messages:
            return old_summary

        # 提取新消息的文本内容
        new_messages_text = ""
        for message in new_messages:
            if isinstance(message,HumanMessage):
                new_messages_text += f"human:{message.content}\n"
            elif isinstance(message,AIMessage):
                new_messages_text += f"assistant:{message.content}\n"

        # 如果没有llm，那么使用简要版本
        if not llm:
            self._simple_incremental_summary(old_summary,new_messages_text)


        # 使用llm生成增量摘要
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",SUMMARY_GENERATION_PROMPT),
                ("human", f"旧摘要：{old_summary}\n新消息：{new_messages_text}")
            ]
        )
        # 生成摘要的过程中出现异常，则使用简要版本
        try:
            summary_chain = summary_prompt|llm|self.str_output_parser
            res = summary_chain.invoke(input={}) # invoke()必须接收一个参数（代表输入）
            new_summary = res.strip()

            if new_summary.startswith("[摘要]"):
                # 7是怎么来的？
                new_summary = new_summary[7:].strip()
            return new_summary
        except Exception as e:
            print(e)
            return self._simple_incremental_summary(old_summary,new_messages_text)


    def _simple_incremental_summary(self,old_summary:str,new_messages_text:str)->str:
        """
        简单的增量摘要：将[旧摘要]和[新消息]合并后，返回合并后的摘要
        :param old_summary: 旧摘要
        :param new_messages_text: 新的会话历史记录
        :return: 增量会话摘要
        """

        if old_summary and old_summary !="[摘要]无":
            # 截取旧摘要的前150个字符+新消息的前150个字符
            old_part = old_summary[:150] if len(old_summary)>=150 else old_summary
            new_part = new_messages_text[:150] if len(new_messages_text)>=150 else new_messages_text
            return f"[摘要] {old_part}...{new_part}"

        return f"[摘要] {new_messages_text}[:150]"


    def _trim_and_summarize(self,llm=None)->None:
        """
        截取会话历史记录，并生成摘要
        :param llm: 语言模型
        :return: 摘要
        """
        # 获取历史消息
        history = self.get_session_history()
        if not history:
            return

        # 未摘要的消息数量
        unsummarized = _unsummarized_message_count.get(self._session_key,0)

        # 达到阈值，触发摘要更新
        if unsummarized>=SUMMARY_BATCH_SIZE:
            # 批量生成摘要
            # 获取所有消息
            messages = history.messages
            total_count = len(messages)
            # 获取已摘要的消息数量
            summarized_count = total_count - unsummarized

            # 获取未摘要的消息（从summarized_count开始）
            new_messages = messages[summarized_count:]
            if new_messages:

                old_summary = _summary_store.get(self._session_key,"")
                # 增量生成摘要信息
                new_summary = self._generate_incremental_summary(old_summary,new_messages,llm)
                # 更新摘要信息
                _summary_store[self._session_key] = new_summary
                # 重置未摘要消息数量
                _unsummarized_message_count[self._session_key] = 0
                print(f"[摘要] {new_summary}")



    def add_user_messages(self,message:str,llm=None)->None:
        """
        添加用户消息到会话历史记录
        :param message: 用户消息
        :return: None
        """

        history = self.get_session_history()
        history.add_user_message(message)

        # 增加未摘要消息数量
        _unsummarized_message_count[self._session_key] += 1
        # 触发摘要更新
        self._trim_and_summarize(llm)


    def add_ai_messages(self,message:str,llm=None)->None:
        """
        添加AI消息到会话历史记录
        :param message: AI消息
        :return: None
        """

        history = self.get_session_history()
        history.add_ai_message(message)

        # 增加未摘要消息数量
        _unsummarized_message_count[self._session_key] += 1
        # 触发摘要更新
        self._trim_and_summarize(llm)


    def prepare_memory_for_llm(self)->List[BaseMessage]:
        """
        为LLM准备记忆
        :return: 记忆字典
        """

        # 获取历史消息
        history = self.get_session_history()
        messages = history.messages

        result = []
        # [关键事实] & [摘要] & [近期会话]
        # 1.添加关键事实
        facts = _fact_store.get(self._session_key,{})
        if facts:
            facts_text = "|".join(f"{key}:{value}" for key,value in facts.items())
            result.append(AIMessage(content=f"关键事实:{facts_text}"))

        # 2.添加摘要
        summary = _summary_store.get(self._session_key,"")
        if summary:
            result.append(AIMessage(content=f"摘要:{summary}"))

        # 3.添加近期原始会话
        # 获取未摘要的消息数量
        unsummarized_count = _unsummarized_message_count.get(self._session_key,0)
        if unsummarized_count>0:
            recent_messages = messages[-unsummarized_count:]
            # result.append(recent_messages) 格式：[... , [msg1, msg2, msg3]]
            # 添加近期原始会话 格式：[... , msg1, msg2, msg3]
            result.extend(recent_messages)

        return result

    def get_key_facts(self)->Dict[str, str]:
        """
        获取当前会话的关键事实
        :return: 关键事实列表
        键为事实名称, 值为事实值
        """

        return _fact_store.get(self._session_key,{}).copy()

    def update_key_facts(self,key_facts:Dict[str, str])->None:
        """
        更新当前会话的关键事实
        :param key_facts: 关键事实字典
        键为事实名称, 值为事实值
        """

        if self._session_key not in _fact_store:
            _fact_store[self._session_key] = {}


        _fact_store[self._session_key].update(
            {k:v for k,v in key_facts.items() if v}
        )

        # print(f"[关键事实] {_fact_store[self._session_key]}")
        print("update key facts")


    def clear_key_facts(self)->None:
        """
        清空当前会话的关键事实
        :return: None
        """

        if self._session_key in _fact_store:
            del _fact_store[self._session_key]































