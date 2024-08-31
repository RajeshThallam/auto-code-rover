"""
For models other than those from OpenAI, use LiteLLM if possible.
"""

import os
import sys
from typing import Literal

import litellm
from litellm.utils import Choices, Message, ModelResponse
from openai import BadRequestError
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_exponential

from app.log import log_and_print
from app.model import common
from app.model.common import Model


class VertexAIModel(Model):
    """
    Base class for creating Singleton instances of Vertex AI Gemini models.
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
        self._initialized = True

    def setup(self) -> None:
        """
        Check API key.
        """
        self.check_api_key()

    def check_api_key(self) -> str:
        project_id = os.getenv("PROJECT_ID")
        if not (project_id):
            print(f"Please set the PROJECT_ID env var")
            sys.exit(1)
            
        import subprocess
        import google.auth

        if __name__ == "__main__":
            subprocess.run(['gcloud', 'config', 'set', 'project', project_id, ])
            credentials, project_id = google.auth.default()
            print(project_id)
        return project_id

    def extract_resp_content(self, chat_message: Message) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
        content = chat_message.content
        if content is None:
            return ""
        else:
            return content

    @retry(wait=wait_exponential(multiplier=1, min=60, max=600), stop=stop_after_attempt(5))
    def call(
        self,
        messages: list[dict],
        top_p=1,
        tools=None,
        response_format: Literal["text", "json_object"] = "text",
        **kwargs,
    ):
        # FIXME: ignore tools field since we don't use tools now
        try:
            prefill_content = "{"
            if response_format == "json_object":  # prefill
                messages.append({"role": "assistant", "content": prefill_content})

            response = litellm.completion(
                model=self.name,
                messages=messages,
                temperature=common.MODEL_TEMP,
                max_tokens=1024,
                top_p=top_p,
                stream=False,
            )
            assert isinstance(response, ModelResponse)
            resp_usage = response.usage
            assert resp_usage is not None
            input_tokens = int(resp_usage.prompt_tokens)
            output_tokens = int(resp_usage.completion_tokens)
            cost = self.calc_cost(input_tokens, output_tokens)

            common.thread_cost.process_cost += cost
            common.thread_cost.process_input_tokens += input_tokens
            common.thread_cost.process_output_tokens += output_tokens

            first_resp_choice = response.choices[0]
            assert isinstance(first_resp_choice, Choices)
            resp_msg: Message = first_resp_choice.message
            content = self.extract_resp_content(resp_msg)
            if response_format == "json_object":
                # prepend the prefilled character
                if not content.startswith(prefill_content):
                    content = prefill_content + content

            return content, cost, input_tokens, output_tokens

        except BadRequestError as e:
            if e.code == "context_length_exceeded":
                log_and_print("Context length exceeded")
            raise e


class GeminiPro(VertexAIModel):
    def __init__(self):
        super().__init__(
            "gemini-1.0-pro-002", 0.00000035, 0.00000105, parallel_tool_call=True
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
            "vertex_ai/gemini-pro-experimental",
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
            "vertex_ai/gemini-flash-experimental",
            0.00000035,
            0.00000105,
            parallel_tool_call=True,
        )
        self.note = "Gemini 1.5 Flash from Google"