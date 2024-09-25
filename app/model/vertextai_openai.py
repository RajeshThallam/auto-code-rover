"""
Interfacing with Vertex AI Gemini models via OpenAI library.
"""
import traceback

import json
import subprocess
import os
import sys
from typing import Literal, cast, Any

from loguru import logger
from openai import BadRequestError, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as OpenaiFunction,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.data_structures import FunctionCallIntent
from app.log import log_and_print
from app.model import common
from app.model.common import Model

import google.auth
import google.auth.transport.requests
import openai

logger.add(sys.stderr, level="DEBUG")
logger.add(sys.stdout, level="DEBUG")

class OpenAICredentialsRefresher:
    def __init__(self, **kwargs: Any) -> None:
        # Set a dummy key here
        self.client = openai.OpenAI(**kwargs, api_key="DUMMY")
        self.creds, self.project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def __getattr__(self, name: str) -> Any:
        if not self.creds.valid:
            auth_req = google.auth.transport.requests.Request()
            self.creds.refresh(auth_req)

            if not self.creds.valid:
                raise RuntimeError("Unable to refresh auth")

            self.client.api_key = self.creds.token
        return getattr(self.client, name)



class VertexAIModel(Model):
    """
    Base class for creating Singleton instances of OpenAI models.
    We use native API from OpenAI instead of LiteLLM.
    """

    _instances = {}

    def __new__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._instances[cls]._initialized = False
        return cls._instances[cls]

    def __init__(
        self,
        name: str,
        cost_per_input: float,
        cost_per_output: float,
        parallel_tool_call: bool = False,
    ):
        if self._initialized:
            return
        super().__init__(name, cost_per_input, cost_per_output, parallel_tool_call)
        # max number of output tokens allowed in model response
        # sometimes we want to set a lower number for models with smaller context window,
        # because output token limit consumes part of the context window
        self.max_output_token = 8000
        # client for making request
        self.client: OpenAI | None = None
        self._initialized = True

    def setup(self) -> None:
        """
        Check API key, and initialize OpenAI client.
        """
        if self.client is None:
            key = self.check_api_key()
            project_id = os.getenv("PROJECT_ID")
            location = "us-central1"
            self.client = OpenAICredentialsRefresher(
                base_url=f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi",
            )


    def check_api_key(self) -> str:
        project_id = os.getenv("PROJECT_ID")
        if not (project_id):
            print(f"Please set the PROJECT_ID env var")
            sys.exit(1)

        if __name__ == "__main__":
            subprocess.run(['gcloud', 'config', 'set', 'project', project_id, ])
            credentials, project_id = google.auth.default()
            print(project_id)
        return project_id

    def extract_resp_content(
        self, chat_completion_message: ChatCompletionMessage
    ) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
        content = chat_completion_message.content
        if content is None:
            return ""
        else:
            return content

    def extract_resp_func_calls(
        self,
        chat_completion_message: ChatCompletionMessage,
    ) -> list[FunctionCallIntent]:
        """
        Given a chat completion message, extract the function calls from it.
        Args:
            chat_completion_message (ChatCompletionMessage): The chat completion message.
        Returns:
            List[FunctionCallIntent]: A list of function calls.
        """
        result = []
        tool_calls = chat_completion_message.tool_calls
        if tool_calls is None:
            return result

        call: ChatCompletionMessageToolCall
        for call in tool_calls:
            called_func: OpenaiFunction = call.function
            func_name = called_func.name
            func_args_str = called_func.arguments
            # maps from arg name to arg value
            if func_args_str == "":
                args_dict = {}
            else:
                try:
                    args_dict = json.loads(func_args_str, strict=False)
                except json.decoder.JSONDecodeError:
                    args_dict = {}
            func_call_intent = FunctionCallIntent(func_name, args_dict, called_func)
            result.append(func_call_intent)

        return result

    # FIXME: the returned type contains OpenAI specific Types, which should be avoided
    @retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
    def call(
        self,
        messages: list[dict],
        top_p: float = 1,
        tools: list[dict] | None = None,
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        **kwargs,
    ) -> tuple[
        str,
        list[ChatCompletionMessageToolCall] | None,
        list[FunctionCallIntent],
        float,
        int,
        int,
    ]:
        """
        Calls the openai API to generate completions for the given inputs.
        Assumption: we only retrieve one choice from the API response.

        Args:
            messages (List): A list of messages.
                            Each item is a dict (e.g. {"role": "user", "content": "Hello, world!"})
            top_p (float): The top_p to use. We usually do not vary this, so not setting it as a cmd-line argument. (from 0 to 1)
            tools (List, optional): A list of tools.

        Returns:
            Raw response and parsed components.
            The raw response is to be sent back as part of the message history.
        """
        print("Making LLM call")
        if temperature is None:
            temperature = common.MODEL_TEMP

        assert self.client is not None
        try:
            if tools is not None and len(tools) == 1:
                # there is only one tool => force the model to use it
                tool_name = tools[0]["function"]["name"]
                tool_choice = {"type": "function", "function": {"name": tool_name}}
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,  # type: ignore
                    tools=tools,  # type: ignore
                    tool_choice=cast(ChatCompletionToolChoiceOptionParam, tool_choice),
                    temperature=temperature,
                    response_format=dict(type=response_format),
                    max_tokens=self.max_output_token,
                    top_p=top_p,
                    stream=False,
                    seed=42,
                )
            else:
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,  # type: ignore
                    tools=tools,  # type: ignore
                    temperature=temperature,
                    response_format=dict(type=response_format), # type: ignore
                    max_tokens=self.max_output_token,
                    top_p=top_p,
                    stream=False,
                    seed=42,
                )

            usage_stats = response.usage
            assert usage_stats is not None

            input_tokens = int(usage_stats.prompt_tokens)
            output_tokens = int(usage_stats.completion_tokens)
            cost = self.calc_cost(input_tokens, output_tokens)

            common.thread_cost.process_cost += cost
            common.thread_cost.process_input_tokens += input_tokens
            common.thread_cost.process_output_tokens += output_tokens

            raw_response = response.choices[0].message
            # log_and_print(f"Raw model response: {raw_response}")
            content = self.extract_resp_content(raw_response)
            raw_tool_calls = raw_response.tool_calls
            func_call_intents = self.extract_resp_func_calls(raw_response)
            return (
                content,
                raw_tool_calls,
                func_call_intents,
                cost,
                input_tokens,
                output_tokens,
            )
        # except BadRequestError as e:
        #     logger.debug("BadRequestError ({}): messages={}", e.code, messages)
        #     if e.code == "context_length_exceeded":
        #         log_and_print("Context length exceeded")
        #     raise e
        except Exception as e:
            # logger.debug("Error ({}): messages={}", e.code, messages)
            # print(repr(e))
            # print("Error ({}): messages={}").format(e.code, messages)
            print(traceback.format_exc())
            raise e
            


class GeminiPro(VertexAIModel):
    def __init__(self):
        super().__init__(
            "gemini-1.0-pro-002", 
            0.00000035, 
            0.00000105, 
            parallel_tool_call=True
        )
        self.note = "Gemini 1.0 from Google"


class Gemini15Pro(VertexAIModel):
    def __init__(self):
        super().__init__(
            "vertex_ai/gemini-1.5-pro-001",
            0.00000035,
            0.00000105,
            parallel_tool_call=True,
        )
        self.note = "Gemini 1.5 from Google"
        
class Gemini15ProExp(VertexAIModel):
    def __init__(self):
        super().__init__(
            "vertex_ai/gemini-1.5-pro-002",
            0.00000035,
            0.00000105,
            parallel_tool_call=True,
        )
        self.note = "Gemini 1.5 from Google"


class Gemini15Flash(VertexAIModel):
    def __init__(self):
        super().__init__(
            "vertex_ai/gemini-1.5-flash-001",
            0.00000035,
            0.00000105,
            parallel_tool_call=True,
        )
        self.note = "Gemini 1.5 Flash from Google"
        
class Gemini15FlashExp(VertexAIModel):
    def __init__(self):
        super().__init__(
            "vertex_ai/gemini-1.5-flash-002",
            0.00000035,
            0.00000105,
            parallel_tool_call=True,
        )
        self.note = "Gemini 1.5 Flash from Google"