# -*- coding: utf-8 -*-
"""
File Name: chat_service.py
Description: 对话service
Author: Hunter
Time: 2026/5/12 22:42
"""

import logging
import time
import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from core.protocol import ChatRequest,ChatResponse
from core.intent import IntentRecognizer
from core.memory import MemoryHandler
from config.settings import DASHSCOPE_API_KEY,TONGYI_MODEL,TONGYI_TEMPERATURE,TONGYI_MAX_TOKEN,CONFIDENCE_THRESHOLD



log = logging.getLogger(__name__)


class ChatService:


    def __init__(self,api_key: str=None):

        # 初始化日志记录器
        if not logging.root.handlers:
            logging.basicConfig(level=logging.INFO)

        # 初始化llm，需要用户传递api_key,否则从环境变量中获取api_key
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise ValueError("DASHSCOPE_API_KEY 环境变量没有设置")

        self._llm = ChatTongyi(
            api_key=DASHSCOPE_API_KEY,
            model=TONGYI_MODEL,
            model_kwargs={"temperature":TONGYI_TEMPERATURE,"max_tokens":TONGYI_MAX_TOKEN}
        )
        self._intent = IntentRecognizer(self._llm)
        # 维护 Memory 对象缓存：key是user_id:session_id
        self._memory_cache:dict[str,MemoryHandler] = {}


    def _get_memory(self,user_id:str,session_id:str):
        """
        获取记忆缓存
        :param user_id: 用户id
        :param session_id: 会话id
        :return:
        """

        key = f"{user_id}:{session_id}"
        if key not in self._memory_cache:
            self._memory_cache[key] = MemoryHandler(user_id,session_id)
        return self._memory_cache[key]

    def handle(self,request:ChatRequest)-> ChatResponse:
        """
        处理用户请求
        :param request: 用户请求
        :return:
        """
        #  请求开始时间
        start_time = time.time()

        try:
            # 1.意图识别
            intent_result = self._intent.recognize(request.user_input)

            # 2.低置信度处理
            if intent_result.confidence < CONFIDENCE_THRESHOLD:
                low_conf = self._handle_low_confidence(intent_result, request.user_input)
                return ChatResponse(
                    message_id=request.message_id,
                    session_id=request.session_id,
                    user_id=request.user_id,
                    response=low_conf,
                    trace_id=request.trace_id
                )

            # 3.确定主要意图
            intents = intent_result.intents
            primary_intent = intents[0] if intents else "general"

            # 4.构建 prompt（不同的意图选择不同的prompt、也就是选择不同的链路）
            prompt = ChatPromptTemplate.from_messages(
                messages=[
                    ("system", self._system_prompt_by_intent(primary_intent)),
                    MessagesPlaceholder("history"),
                    ("human", "用户输入:{user_input}")
                ])
            # 5.记忆管理
            memory = self._get_memory(request.user_id, request.session_id)

            def prepare(input_dict: dict)->dict:
                # 准备提供给大语言模型的上下文
                memory_messages = memory.prepare_memory_for_llm()
                return {
                    "user_input": input_dict["user_input"],
                    "history": memory_messages
                }

            # 6.调用llm(记忆准备->Memory)
            chain = RunnableLambda(prepare)|prompt|self._llm|StrOutputParser()
            res = chain.invoke(input={"user_input": request.user_input})
            # 7.写回历史消息
            memory.add_user_messages(request.user_input,self._llm)
            memory.add_ai_messages(res,self._llm)
            # 8.更新事实
            memory.update_key_facts(intent_result.slots)
            # 9.日志记录
            self._log(req=request,
                intents=intents,
                confidence=intent_result.confidence,
                action="normal",
                latency_ms=int((time.time() - start_time) * 1000)
                )
            return ChatResponse(
                message_id=request.message_id,
                session_id=request.session_id,
                user_id=request.user_id,
                response=res,
                trace_id=request.trace_id
            )


        except Exception as e:
            self._log(
                req=request,
                intents=["error"],
                confidence=0.0,
                action="error",
                latency_ms=int((time.time() - start_time) * 1000),
                error=str(e),
            )
            raise e

    def _handle_low_confidence(self,intent_result, user_input: str) -> str|None:
        return "抱歉，我没太理解您的需求。请问您是想：\n" \
               "1️⃣ 查询物流进度\n" \
               "2️⃣ 修改收货地址\n" \
               "3️⃣ 申请退货退款\n" \
               "4️⃣ 投诉建议\n" \
               "请回复数字或具体需求。"

    def _system_prompt_by_intent(self,intent: str) -> str:
        mapping = {
        "track_shipping": "你是电商物流查询客服。先要订单号/运单号；如果缺失就追问。",
        "change_address": "你是电商改地址客服。先确认是否已发货；需要订单号+新地址。",
        "refund": "你是电商退货退款客服。先给3步结论，再给注意事项，最后引导用户提供订单号。",
        "complaint": "你是电商投诉处理客服。先安抚，再收集订单号与问题细节，给出处理时效。",
        "general": "你是一个友好、简洁的聊天助手。",
        }
        return mapping.get(intent.strip(), mapping["general"])

    def _log(self,
              *,
              req: ChatRequest,
              intents: list[str],
              confidence: float,
              action: str,
              latency_ms: int,
              error: str | None = None):
        """

        :param req: 请求
        :param intents: 意图
        :param confidence: 置信度
        :param action: 动作
        :param latency_ms: 花费时间
        :param error: 错误
        :return:
        """
        payload = {
            "trace_id": req.trace_id,
            "user_id": req.user_id,
            "session_id": req.session_id,
            "message_id": req.message_id,
            "intents": intents,
            "confidence": confidence,
            "action": action,
            "latency_ms": latency_ms
        }
        # 如果有错误信息，记录为错误日志error，否则记录为普通日志info
        if error:
            payload["error"] = error
            log.error(payload)
        else:
            log.info(payload)





























