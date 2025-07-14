import time

from colorama import Fore, Style

def fancy_print(message: str) -> None:
    """
    Displays a fancy print message.

    Args:
        message (str): The message to display.
    """
    print(Style.BRIGHT + Fore.CYAN + f"\n{'=' * 50}")
    print(Fore.MAGENTA + f"{message}")
    print(Style.BRIGHT + Fore.CYAN + f"{'=' * 50}\n")
    time.sleep(0.5)

def fancy_step_tracker(step: int, total_steps: int) -> None:
    """
    Displays a fancy step tracker.

    Args:
        step (int): The current step.
        total (int): The total number of steps.
    """
    fancy_print(f"STEP {step + 1}/{total_steps}")
