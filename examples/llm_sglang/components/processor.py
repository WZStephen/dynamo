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
import uuid
from enum import Enum
from typing import AsyncIterator, Any, Tuple, Union

from components.kv_router import Router
from components.worker import SGLangWorker
from transformers import AutoTokenizer
from utils.chat_processor import ChatProcessor, CompletionsProcessor, ProcessMixIn
from utils.protocol import MyRequestOutput, Tokens, vLLMGenerateRequest
from utils.vllm import parse_vllm_args
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from vllm.logger import logger as vllm_logger
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer

from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

import json
from utils.sglang import parse_sglang_args
from sglang.srt.openai_api.protocol import ChatCompletionRequest, CompletionRequest
from sglang.srt.openai_api.adapter import (
    v1_batches,
    v1_cancel_batch,
    v1_chat_completions,
    v1_completions,
    v1_delete_file,
    v1_embeddings,
    v1_files_create,
    v1_retrieve_batch,
    v1_retrieve_file,
    v1_retrieve_file_content,
)
from sglang.srt.openai_api.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    ChatCompletionStreamResponse,
    ChatCompletionResponse,
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


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Processor(ProcessMixIn):
    """
    vLLM pre and post processing
    """

    worker = depends(SGLangWorker)
    router = depends(Router)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_sglang_args(class_name, "")
        # self.model_config = self.engine_args.create_model_config()
        # self.chat_processor = ChatProcessor(self.tokenizer, self.model_config)
        # self.completions_processor = CompletionsProcessor(
        #     self.tokenizer, self.model_config
        # )
        self.min_workers = 1

        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",
                truncation_side="left",
                use_fast=True,
            )
        except Exception as e:
            print(f"Failed to load tokenizer: {e}. Using default processing.")
            self.tokenizer = None

        self.router_mode = self.engine_args.router

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = SGLangWorker.dynamo_address()  # type: ignore
        print(f"[Processor] initializing SGLangWorker with runtime: {runtime}, {comp_ns}, {comp_name}")
        self.worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )
        while len(self.worker_client.endpoint_ids()) < self.min_workers:
            print(
                f"Waiting for workers to be ready.\n"
                f" Current: {len(self.worker_client.endpoint_ids())},"
                f" Required: {self.min_workers}"
            )
            await asyncio.sleep(2)
        print(f"[Processor] Workers are ready, {self.worker_client.endpoint_ids()}")

    async def _generate(
        self,
        raw_request: ChatCompletionRequest,
        request_type: RequestType,
    ):
        print(f"[Processor] processor got request: {raw_request}")
        # sampling_params = json.dumps({"temperature": 1, "top_p": 1})
        engine_generator = await self.worker_client.generate(
            raw_request.model_dump_json()
        )
        async for response in engine_generator:
            # receive response from engine
            response_data = response.data()
            # deserialize response to ChatCompletionStreamResponse
            response = json.loads(response_data)
            # convert empty strings to None
            def replace_empty_with_none(obj):
                if isinstance(obj, dict):
                    return {k: replace_empty_with_none(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_empty_with_none(i) for i in obj]
                elif obj == "":
                    return None
                else:
                    return obj
            response = replace_empty_with_none(response)
            print(f"[Processor] engine_generator: {response}")
            yield response

    @dynamo_endpoint(name="chat/completions")
    async def chat_completions(self, raw_request: ChatCompletionRequest):
        async for response in self._generate(raw_request, RequestType.CHAT):
            yield response

    # @dynamo_endpoint()
    # async def completions(self, raw_request: CompletionRequest):
    #     async for response in self._generate(raw_request, RequestType.COMPLETION):
    #         yield response


# INFO 04-08 09:48:26 engine.py:404] Added request c9e78146-cb00-401d-8b4e-b2099ed7db54.
# INFO 04-08 09:48:27 metrics.py:455] Avg prompt throughput: 0.6 tokens/s, Avg generation throughput: 0.1 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
# INFO 04-08 09:48:27 metrics.py:471] Prefix cache hit rate: GPU: 0.00%, CPU: 0.00%
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'role': 'assistant'ntent': 'Okay'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'co, 'content': ''}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ','}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' the'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' user'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' typed'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' "'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': 'Hi'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': '".'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' That'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': "'s"}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' a'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' friendly'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' greeting'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': '.\n\n'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': 'I'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' should'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' respond'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' in'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' a'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' welcoming'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' manner'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': '.\n\n'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': 'Maybe'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' say'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' hello'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' and'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' ask'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' how'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' I'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' can'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' assist'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' them'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' today'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': '.\n\n'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': 'Keep'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' it'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' open'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': '-ended'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' to'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' encourage'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' them'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' to'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' share'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' what'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' they'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' need'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' help'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' with'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': '.\n'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': '</think>'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': '\n\n'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': 'Hello'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': '!'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' How'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' can'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' I'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' assist'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' you'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ' today'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': '?'}, 'logprobs': None, 'finish_reason': None}]}
# {'id': 'c9e78146-cb00-401d-8b4e-b2099ed7db54', 'object': 'chat.completion.chunk', 'created': 1744105706, 'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'choices': [{'index': 0, 'delta': {'content': ''}, 'logprobs': None, 'finish_reason': 'stop', 'stop_reason': None}]}
# INFO 04-08 09:48:40 metrics.py:455] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4.5 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
# INFO 04-08 09:48:40 metrics.py:471] Prefix cache hit rate: GPU: 0.00%, CPU: 0.00%
# INFO 04-08 09:48:50 metrics.py:455] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
