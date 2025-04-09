# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import os

from components.disagg_router import PyDisaggregatedRouter
from components.prefill_worker import PrefillWorker
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.protocol import MyRequestOutput
from utils.sglang import parse_sglang_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.logger import logger as vllm_logger
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest
from vllm.sampling_params import RequestOutputKind

from dynamo.llm import KvMetricsPublisher
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

from typing import Any, Dict, List, Optional, Union, AsyncIterator, Callable
from http import HTTPStatus
from transformers import AutoTokenizer
import dataclasses
import uvloop
import json
import asyncio
import sglang as sgl
from sglang.srt.openai_api.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    ChatCompletionStreamResponse,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    LogProbs,
    TopLogprob,
    ChatCompletionTokenLogprob,
    ChoiceLogprobs,
    FunctionResponse,
    ToolCall,
    UsageInfo,
)
from sglang.srt.function_call_parser import TOOLS_TAG_LIST, FunctionCallParser
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api.adapter import (
    v1_chat_generate_request,
    v1_chat_generate_response,
    create_error_response
)
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.server_args import ServerArgs
from fastapi.responses import ORJSONResponse, StreamingResponse
import time
from utils.sglang import v1_chat_completions
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: TokenizerManager
    scheduler_info: Dict


_global_state: Optional[_GlobalState] = None


def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state


def to_openai_style_logprobs(
    input_token_logprobs=None,
    output_token_logprobs=None,
    input_top_logprobs=None,
    output_top_logprobs=None,
):
    ret_logprobs = LogProbs()

    def append_token_logprobs(token_logprobs):
        for logprob, _, token_text in token_logprobs:
            ret_logprobs.tokens.append(token_text)
            ret_logprobs.token_logprobs.append(logprob)

            # Not supported yet
            ret_logprobs.text_offset.append(-1)

    def append_top_logprobs(top_logprobs):
        for tokens in top_logprobs:
            if tokens is not None:
                ret_logprobs.top_logprobs.append(
                    {token[2]: token[0] for token in tokens}
                )
            else:
                ret_logprobs.top_logprobs.append(None)

    if input_token_logprobs is not None:
        append_token_logprobs(input_token_logprobs)
    if output_token_logprobs is not None:
        append_token_logprobs(output_token_logprobs)
    if input_top_logprobs is not None:
        append_top_logprobs(input_top_logprobs)
    if output_top_logprobs is not None:
        append_top_logprobs(output_top_logprobs)

    return ret_logprobs


def create_streaming_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> str:
    error = ErrorResponse(message=message, type=err_type, code=status_code.value)
    json_str = json.dumps({"error": error.model_dump()})
    return json_str


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class SGLangWorker:
    prefill_worker = depends(PrefillWorker)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_sglang_args(class_name, "")
        print(f"[Worker] engine_args: {self.engine_args}")
        self.model_name = (
            self.engine_args.served_model_name
            if self.engine_args.served_model_name is not None
            else "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        )
        self.tensor_parallel_size = self.engine_args.tensor_parallel_size

        # Disaggregated routing configuration
        self.do_remote_prefill = self.engine_args.remote_prefill
        self.conditional_disagg = self.engine_args.conditional_disagg
        self.max_local_prefill_length = self.engine_args.max_local_prefill_length
        self.max_prefill_queue_size = self.engine_args.max_prefill_queue_size

        # Prefill queue configuration
        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )

        # Initialize metrics publisher
        self.metrics_publisher = KvMetricsPublisher()

        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception as e:
            print(f"[Worker] Error loading tokenizer: {e}")
            self.tokenizer = None

        # Initialize disaggregated router
        if self.conditional_disagg:
            self.disaggregated_router = PyDisaggregatedRouter(
                dynamo_context["runtime"],
                self.model_name,
                max_local_prefill_length=self.max_local_prefill_length,
                max_prefill_queue_size=self.max_prefill_queue_size,
            )
        else:
            self.disaggregated_router = None

        # Initialize runtime
        self.engine = None

    @async_on_start
    async def async_init(self):
        # self.engine = InferenceEngine(
        #     model_path=self.model_name,
        # )
        server_args = ServerArgs(
            model_path=self.model_name,
        )
        tokenizer_manager, scheduler_info = _launch_subprocesses(server_args=server_args)
        set_global_state(
            _GlobalState(
                tokenizer_manager=tokenizer_manager,
                scheduler_info=scheduler_info,
            )
        )
        test_prompt = "Who are you?"

        # 构建 raw_request，将请求包装成一个列表
        raw_request = [ChatCompletionRequest(  # 注意这里是一个列表
            model=self.model_name,
            messages=[{"role": "user", "content": test_prompt}],
        )]

        # 调用 v1_chat_generate_request，获取返回的适配后的请求和实际请求
        adapted_request, request = v1_chat_generate_request(raw_request, _global_state.tokenizer_manager)

        print(f"[Worker] Adapted request: {adapted_request}")
        print(f"[Worker] Request: {request}")

        try:
            ret = await tokenizer_manager.generate_request(
                adapted_request
            ).__anext__()
        except ValueError as e:
            return create_error_response(str(e))
        if not isinstance(ret, list):
            ret = [ret]

        response = v1_chat_generate_response(
            request=request,
            ret=ret
        )

        print(f"[Worker] Test Generated Response: {response}")

        self.disaggregated_router = None
        print("[Worker] SGLangWorker has been initialized")

    async def create_metrics_publisher_endpoint(self):
        component = dynamo_context["component"]
        await self.metrics_publisher.create_endpoint(component)

    def get_remote_prefill_request_callback(self):
        # TODO: integrate prefill_queue to dynamo endpoint
        async def callback(request: RemotePrefillRequest):
            async with PrefillQueue.get_instance(
                nats_server=self._prefill_queue_nats_server,
                stream_name=self._prefill_queue_stream_name,
            ) as prefill_queue:
                await prefill_queue.enqueue_prefill_request(request)

        return callback

    @dynamo_endpoint()
    async def generate(self, request: ChatCompletionRequest):
        print(f"[Worker] worker received request: {request}")

        tokenizer_manager = _global_state.tokenizer_manager
        async for response in v1_chat_completions(tokenizer_manager, request):
            print(f"[worker] {response}")
            yield response
