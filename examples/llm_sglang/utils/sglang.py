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

import argparse
import os
import json
import uuid
from argparse import Namespace

from dynamo.sdk.lib.config import ServiceConfig
from typing import AsyncGenerator

from sglang.utils import get_exception_traceback
from sglang.srt.function_call_parser import FunctionCallParser
from sglang.srt.openai_api.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    FileDeleteResponse,
    FileRequest,
    FileResponse,
    FunctionResponse,
    LogProbs,
    ToolCall,
    TopLogprob,
    UsageInfo,
)
from sglang.srt.openai_api.adapter import (
    to_openai_style_logprobs,
    v1_chat_generate_request,
    v1_chat_generate_response,
    create_streaming_error_response,
    create_error_response
)


def parse_sglang_args(service_name, model_path) -> Namespace:
    """Parse arguments for SGLang engine"""
    parser = argparse.ArgumentParser()
    
    # Model configuration
    parser.add_argument(
        "--model", type=str, default=model_path, help="Model to use for generation"
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Name of the model being served",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism",
    )
    
    # Router configuration
    parser.add_argument(
        "--router",
        type=str,
        default="random",
        choices=["random", "round-robin", "kv"],
        help="Router to use for request distribution",
    )
    
    # Prefill configuration
    parser.add_argument(
        "--remote-prefill",
        action="store_true",
        help="Enable remote prefill",
    )
    parser.add_argument(
        "--conditional-disagg",
        action="store_true",
        help="Enable conditional disaggregation",
    )
    parser.add_argument(
        "--max-local-prefill-length",
        type=int,
        default=1000,
        help="Maximum length of prefill that can be done locally",
    )
    parser.add_argument(
        "--max-prefill-queue-size",
        type=int,
        default=2,
        help="Maximum size of the prefill queue",
    )
    
    # SGLang specific configuration
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size for SGLang",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=16384,
        help="Maximum batch tokens for SGLang",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[None, "int8", "int4", "nf4"],
        help="Quantization method for SGLang",
    )
    parser.add_argument(
        "--enable-lora",
        action="store_true",
        help="Enable LoRA support",
    )
    parser.add_argument(
        "--enable-vllm-compatibility",
        action="store_true",
        help="Enable compatibility mode with vLLM API",
    )
    
    # Parse arguments from config
    config = ServiceConfig.get_instance()
    config_args = config.as_args(service_name, prefix="")
    args = parser.parse_args(config_args)
    
    # Override with environment variables if present
    if os.environ.get("SGLANG_MODEL"):
        args.model = os.environ.get("SGLANG_MODEL")
    if os.environ.get("SGLANG_TENSOR_PARALLEL_SIZE"):
        args.tensor_parallel_size = int(os.environ.get("SGLANG_TENSOR_PARALLEL_SIZE"))
    if os.environ.get("SGLANG_MAX_TOKENS"):
        args.max_tokens = int(os.environ.get("SGLANG_MAX_TOKENS"))
    
    return args

async def v1_chat_completions(
    tokenizer_manager, raw_request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    # request_json = await raw_request.json()
    all_requests = [raw_request]
    adapted_request, request = v1_chat_generate_request(all_requests, tokenizer_manager)
    # adapted_request.stream = False
    # 如果是流式请求
    parser_dict = {}

    async def generate_stream_resp():
        is_firsts = {}
        stream_buffers = {}
        n_prev_tokens = {}
        prompt_tokens = {}
        completion_tokens = {}
        try:
            async for content in tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                index = content.get("index", 0)
                text = content["text"]

                is_first = is_firsts.get(index, True)
                stream_buffer = stream_buffers.get(index, "")
                n_prev_token = n_prev_tokens.get(index, 0)

                prompt_tokens[index] = content["meta_info"]["prompt_tokens"]
                completion_tokens[index] = content["meta_info"]["completion_tokens"]
                if request.logprobs:
                    logprobs = to_openai_style_logprobs(
                        output_token_logprobs=content["meta_info"][
                            "output_token_logprobs"
                        ][n_prev_token:],
                        output_top_logprobs=content["meta_info"][
                            "output_top_logprobs"
                        ][n_prev_token:],
                    )

                    n_prev_token = len(
                        content["meta_info"]["output_token_logprobs"]
                    )
                    token_logprobs = []
                    for token, logprob in zip(
                        logprobs.tokens, logprobs.token_logprobs
                    ):
                        token_bytes = list(token.encode("utf-8"))
                        top_logprobs = []
                        if logprobs.top_logprobs:
                            for top_token, top_logprob in logprobs.top_logprobs[
                                0
                            ].items():
                                top_token_bytes = list(top_token.encode("utf-8"))
                                top_logprobs.append(
                                    TopLogprob(
                                        token=top_token,
                                        bytes=top_token_bytes,
                                        logprob=top_logprob,
                                    )
                                )
                        token_logprobs.append(
                            ChatCompletionTokenLogprob(
                                token=token,
                                bytes=token_bytes,
                                logprob=logprob,
                                top_logprobs=top_logprobs,
                            )
                        )

                    choice_logprobs = ChoiceLogprobs(content=token_logprobs)

                else:
                    choice_logprobs = None

                finish_reason = content["meta_info"]["finish_reason"]
                # 判断是否是第一次，构建响应数据
                if is_first:
                    # First chunk with role
                    is_first = False
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(role="assistant", content=""),
                        finish_reason=(
                            finish_reason["type"] if finish_reason else ""
                        ),
                        matched_stop=(
                            finish_reason["matched"]
                            if finish_reason and "matched" in finish_reason
                            else None
                        ),
                        logprobs=choice_logprobs,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield chunk.model_dump_json()

                text = content["text"]
                delta = text[len(stream_buffer) :]
                new_stream_buffer = stream_buffer + delta
                # 如果有工具调用
                if request.tool_choice != "none" and request.tools:
                    if index not in parser_dict:
                        parser_dict[index] = FunctionCallParser(
                            tools=request.tools,
                            tool_call_parser=tokenizer_manager.server_args.tool_call_parser,
                        )
                    parser = parser_dict[index]

                    # parse_increment => returns (normal_text, calls)
                    normal_text, calls = parser.parse_stream_chunk(delta)

                    # 1) if there's normal_text, output it as normal content
                    if normal_text:
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(content=normal_text),
                            finish_reason=(
                                finish_reason["type"] if finish_reason else ""
                            ),
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            choices=[choice_data],
                            model=request.model,
                        )
                        yield chunk.model_dump_json()

                    # 2) if we found calls, we output them as separate chunk(s)
                    for call_item in calls:
                        # transform call_item -> FunctionResponse + ToolCall

                        if (
                            content["meta_info"]["finish_reason"]
                            and content["meta_info"]["finish_reason"]["type"]
                            == "stop"
                        ):
                            latest_delta_len = 0
                            if isinstance(call_item.parameters, str):
                                latest_delta_len = len(call_item.parameters)

                            expected_call = json.dumps(
                                parser.multi_format_parser.detectors[0]
                                .prev_tool_call_arr[index]
                                .get("arguments", {}),
                                ensure_ascii=False,
                            )
                            actual_call = parser.multi_format_parser.detectors[
                                0
                            ].streamed_args_for_tool[index]
                            if latest_delta_len > 0:
                                actual_call = actual_call[:-latest_delta_len]
                            remaining_call = expected_call.replace(
                                actual_call, "", 1
                            )
                            call_item.parameters = remaining_call

                        tool_call = ToolCall(
                            id=str(call_item.tool_index),
                            function=FunctionResponse(
                                name=call_item.name,
                                arguments=call_item.parameters,
                            ),
                        )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(
                                role="assistant", tool_calls=[tool_call]
                            ),
                            finish_reason="tool_call",
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            choices=[choice_data],
                            model=request.model,
                        )
                        yield chunk.model_dump_json()

                    stream_buffers[index] = new_stream_buffer
                    is_firsts[index] = is_first
                # 普通文本的处理
                else:
                    # No tool calls => just treat this as normal text
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(content=delta),
                        finish_reason=(
                            finish_reason["type"] if finish_reason else ""
                        ),
                        matched_stop=(
                            finish_reason["matched"]
                            if finish_reason and "matched" in finish_reason
                            else None
                        ),
                        logprobs=choice_logprobs,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield chunk.model_dump_json()
                    # 更新缓冲区
                    stream_buffers[index] = new_stream_buffer
                    is_firsts[index] = is_first
            # 如果需要返回使用情况
            if request.stream_options and request.stream_options.include_usage:
                total_prompt_tokens = sum(
                    tokens
                    for i, tokens in prompt_tokens.items()
                    if i % request.n == 0
                )
                total_completion_tokens = sum(
                    tokens for tokens in completion_tokens.values()
                )
                usage = UsageInfo(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=str(uuid.uuid4().hex),
                    choices=[],
                    model=request.model,
                    usage=usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}"
        except ValueError as e:
            error = create_streaming_error_response(str(e))
            yield f"data: {error}"
        # yield "data: [DONE]"

    # return the async generator
    async for response in generate_stream_resp():
        yield response



