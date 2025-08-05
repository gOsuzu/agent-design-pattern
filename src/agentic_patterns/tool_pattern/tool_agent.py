import json
import re

from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from agentic_patterns.tool_pattern.tool import Tool
from agentic_patterns.tool_pattern.tool import validate_arguments
from agentic_patterns.utils.completions import build_prompt_structure
from agentic_patterns.utils.completions import ChatHistory
from agentic_patterns.utils.completions import completions_create
from agentic_patterns.utils.completions import update_chat_history
from agentic_patterns.utils.extraction import extract_tag_content

load_dotenv()


TOOL_SYSTEM_PROMPT = """
You are a helpful AI assistant that can use tools when appropriate. You are provided with function signatures within <tools></tools> XML tags.

IMPORTANT RULES:
1. Only call tools when they are specifically needed to answer the user's question
2. If the user asks a general question that doesn't require external data or specific tools, respond directly without calling any tools
3. Examples of questions that DON'T need tools:
   - "What's your name?" → Respond directly: "I'm an AI assistant..."
   - "How are you?" → Respond directly: "I'm doing well, thank you..."
   - "What can you do?" → Respond directly: "I can help you with..."
4. Examples of questions that DO need tools:
   - "Get me the latest news" → Use news-related tools
   - "What's the weather like?" → Use weather tools
   - "Search for information about..." → Use search tools
5. If you decide to call a tool, make sure the function name exactly matches one of the available tools
6. Never use null, None, or empty string as function names

When you need to call a function, return a json object with function name and arguments within <tool_call></tool_call> XML tags:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>,  "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools:

<tools>
%s
</tools>
"""


class ToolAgent:
    """
    The ToolAgent class represents an agent that can interact with a language model and use tools
    to assist with user queries. It generates function calls based on user input, validates arguments,
    and runs the respective tools.

    Attributes:
        tools (Tool | list[Tool]): A list of tools available to the agent.
        model (str): The model to be used for generating tool calls and responses.
        client (Groq): The Groq client used to interact with the language model.
        tools_dict (dict): A dictionary mapping tool names to their corresponding Tool objects.
    """

    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self.client = Groq()
        self.model = model
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def add_tool_signatures(self) -> str:
        """
        Collects the function signatures of all available tools.

        Returns:
            str: A concatenated string of all tool function signatures in JSON format.
        """
        return "".join([tool.fn_signature for tool in self.tools])

    def process_tool_calls(self, tool_calls_content: list) -> dict:
        """
        Processes each tool call, validates arguments, executes the tools, and collects results.

        Args:
            tool_calls_content (list): List of strings, each representing a tool call in JSON format.

        Returns:
            dict: A dictionary where the keys are tool call IDs and values are the results from the tools.
        """
        observations = {}
        for tool_call_str in tool_calls_content:
            try:
                tool_call = json.loads(tool_call_str)
                tool_name = tool_call["name"]
                
                # ツール名がNoneまたは空の場合のエラーハンドリング
                if tool_name is None or tool_name == "":
                    print(Fore.RED + f"\nError: Invalid tool name '{tool_name}'")
                    observations[f"error_{len(observations)}"] = f"Invalid tool name: {tool_name}"
                    continue
                
                # ツールが存在しない場合のエラーハンドリング
                if tool_name not in self.tools_dict:
                    print(Fore.RED + f"\nError: Tool '{tool_name}' not found. Available tools: {list(self.tools_dict.keys())}")
                    observations[f"error_{len(observations)}"] = f"Tool '{tool_name}' not found. Available tools: {list(self.tools_dict.keys())}"
                    continue
                
                tool = self.tools_dict[tool_name]

                print(Fore.GREEN + f"\nUsing Tool: {tool_name}")

                # Validate and execute the tool call
                validated_tool_call = validate_arguments(
                    tool_call, json.loads(tool.fn_signature)
                )
                print(Fore.GREEN + f"\nTool call dict: \n{validated_tool_call}")

                result = tool.run(**validated_tool_call["arguments"])
                print(Fore.GREEN + f"\nTool result: \n{result}")

                # Store the result using the tool call ID
                observations[validated_tool_call["id"]] = result
                
            except json.JSONDecodeError as e:
                print(Fore.RED + f"\nError: Invalid JSON in tool call: {e}")
                observations[f"error_{len(observations)}"] = f"Invalid JSON in tool call: {e}"
            except KeyError as e:
                print(Fore.RED + f"\nError: Missing key in tool call: {e}")
                observations[f"error_{len(observations)}"] = f"Missing key in tool call: {e}"
            except Exception as e:
                print(Fore.RED + f"\nError: Unexpected error in tool call: {e}")
                observations[f"error_{len(observations)}"] = f"Unexpected error: {e}"

        return observations

    def run(
        self,
        user_msg: str,
    ) -> str:
        """
        Handles the full process of interacting with the language model and executing a tool based on user input.

        Args:
            user_msg (str): The user's message that prompts the tool agent to act.

        Returns:
            str: The final output after executing the tool and generating a response from the model.
        """
        user_prompt = build_prompt_structure(prompt=user_msg, role="user")

        tool_chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=TOOL_SYSTEM_PROMPT % self.add_tool_signatures(),
                    role="system",
                ),
                user_prompt,
            ]
        )
        agent_chat_history = ChatHistory([user_prompt])

        tool_call_response = completions_create(
            self.client, messages=tool_chat_history, model=self.model
        )
        tool_calls = extract_tag_content(str(tool_call_response), "tool_call")

        if tool_calls.found:
            observations = self.process_tool_calls(tool_calls.content)
            update_chat_history(
                agent_chat_history, f'f"Observation: {observations}"', "user"
            )

        return completions_create(self.client, agent_chat_history, self.model)
