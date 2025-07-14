from turtle import update
from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from agentic_patterns.utils.completions import build_prompt_structure
from agentic_patterns.utils.completions import completions_create
from agentic_patterns.utils.completions import FixedFirstChatHistory
from agentic_patterns.utils.completions import update_chat_history
from agentic_patterns.utils.logging import fancy_step_tracker

load_dotenv()

BASE_GENERATION_SYSTEM_PROMPT = """
Your task is to Generate the best content possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempt.
You must always output the revised content.
"""

BASE_REFLECTION_SYSTEM_PROMPT = """
You are tasked with generating critique and recommendations to the user's generated content.
If the user content has something wrong or something to be improved, output a list of recommendations
and critiques. If the user content is ok and there's nothing to change, output this: <OK>
"""

class ReflectionAgent:
    """
    A class that represents a reflection agent.The agent first generates
    responses based on provided prompts and then critiques them in a reflection step.

    Attributes:
        model (str): The model name used for generating and reflecting on responses.
        client (Groq): An instance of the Groq client to interact with the language model.
    """
    def __init__(self, model: str = "qwen/qwen3-32b"):
        self.client = Groq()
        self.model = model
    
    def _request_completion(
        self, 
        history: list,
        verbose: int=0,
        log_title: str = "COMPLETION",
        log_color: str=""):
        """
        Request a completion from the language model.

        Args:
            history (list): A list of messages forming the conversation or reflection history.
            verbose (int, optional): The verbosity level. Defaults to 0 (no output).
        """
        output = completions_create(self.client, history, self.model)
        if verbose > 0:
            print(log_color, f"\n\n{log_title}\n\n", output)
        return output

    def generate(self, generation_history: list, verbose: int=0) -> str:
        """
        Generate a response based on the provided generation history.

        Args:
            generation_history (list): A list of messages forming the conversation or reflection history.
            verbose (int, optional): The verbosity level. Defaults to 0 (no output).
        
        Returns:
            str: The generated response.
        """
        return self._request_completion(
            generation_history, verbose, log_title="GENERATION", log_color=Fore.BLUE)

    def reflect(self, reflection_history: list, verbose: int=0) -> str:
        """
        Reflect on the provided reflection history.

        Args:
            reflection_history (list): A list of messages forming the conversation or reflection history.
            verbose (int, optional): The verbosity level. Defaults to 0 (no output).
        """
        return self._request_completion(
            reflection_history, verbose, log_title="REFLECTION", log_color=Fore.GREEN)

    def run(
        self, 
        user_msg: str,
        generation_system_prompt: str="",
        reflection_system_prompt: str="",
        n_steps: int=10,
        verbose: int=0,
    ) -> str:
        """
        Run the ReflectionAgent over multiple steps.
        Generating a response and reflecting on it for the specified times of steps.

        Args:
            user_msg (str): The user's message.
            generation_system_prompt (str, optional): The system prompt for generation. Defaults to BASE_GENERATION_SYSTEM_PROMPT.
            reflection_system_prompt (str, optional): The system prompt for reflection. Defaults to BASE_REFLECTION_SYSTEM_PROMPT.
            m_steps (int, optional): The number of steps to run. Defaults to 10.
            verbose (int, optional): The verbosity level. Defaults to 0 (no output).

        Returns:
            str: The final response after reflection
        """
        generation_system_prompt += BASE_GENERATION_SYSTEM_PROMPT
        reflection_system_prompt += BASE_REFLECTION_SYSTEM_PROMPT

        generation_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=generation_system_prompt, role="system"),
                build_prompt_structure(prompt=user_msg, role="user"),
            ],
            total_length=3,
        )

        reflection_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=reflection_system_prompt, role="system"),
            ],
            total_length=3,
        )

        for step in range(n_steps):
            if verbose > 0:
                fancy_step_tracker(step, n_steps)
            
            # Generate a response
            generation = self.generate(generation_history, verbose=verbose)
            update_chat_history(generation_history, generation, role="assistant")
            update_chat_history(reflection_history, generation, role="user")

            # Reflect and critique the generation response
            critique = self.reflect(reflection_history, verbose=verbose)

            if "<OK>" in critique:
                print(
                    Fore.RED,
                    "\n\nThe user's content is ok and there's nothing to change.\n\n",
                )
                break

            # Update the reflection history with the critique
            update_chat_history(reflection_history, critique, role="user")
            update_chat_history(reflection_history, critique, role="assistant")


        return generation