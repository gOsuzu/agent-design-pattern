def completions_create(client, messages: list[dict], model:str) -> str:
    """
    Creates a completion for a given client, messages, and model.

    Args:
        client(Groq): The client to use for the completion.
        messages (list[dict]): A list of message objects containing chat history for the model.
        model: The model to use for the completion.
    """
    response = client.chat.completions.create(messages=messages, model=model)
    return str(response.choices[0].message.content)


def build_prompt_structure(prompt: str, role: str, tag: str = "") -> dict:
    """
    Builds a prompt structure for a given prompt, role, and tag.

    Args:
        prompt (str): The prompt to build the structure for.
        role (str): The role of the prompt.
        tag (str): The tag of the prompt.

    Returns:
        dict: A dictionary containing the prompt structure.
    """
    if tag:
        prompt = f"<{tag}>{prompt}</{tag}>"
    return {"role": role, "content": prompt}


def update_chat_history(history: list, msg: str, role: str):
    """
    Updates the chat history with a new message.

    Args:
        history (list): The chat history to update.
        msg (str): The message to add to the chat history.
        role (str): The role of the message.
    """
    history.append(build_prompt_structure(prompt=msg, role=role))


class ChatHistory(list):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """
        Initialize the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The total length of the chat history.
        """
        if messages is None:
            messages = []

        super().__init__(messages)
        self.total_length = total_length

    def append(self, msg: str):
        """
        Add a message to the chat history.

        Args:
            msg (str): The message to add to the chat history.
        """
        if len(self) == self.total_length:
            self.pop(0)
        super().append(msg)


class FixedFirstChatHistory(ChatHistory):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """
        Initialize the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The total length of the chat history.
        """
        super().__init__(messages, total_length)

    def append(self, msg: str):
        """
        Add a message to the chat history. The first message will always stay fixed.

        Args:
            msg (str): The message to add to the chat history.
        """
        if len(self) == self.total_length:
            self.pop(1)
        super().append(msg)

