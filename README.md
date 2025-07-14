# agent-design-pattern

This is basic 4 design patterns of LLM agent: reflection, tool-usage, planning and multi-agent pattern.

## Installation

```bash
# Install the package in development mode
uv pip install -e .
```

## Reflection Pattern

The Reflection Pattern is an agent design pattern that implements a self-improving loop where the agent generates responses and then critiques them to improve quality.

### How it Works

1. **Generation Phase**: The agent generates an initial response to the user's request
2. **Reflection Phase**: The agent critiques its own response and identifies areas for improvement
3. **Iteration**: The process repeats until the agent determines the response is satisfactory (`<OK>`)

### Usage

```python
from agentic_patterns import ReflectionAgent

# Create a reflection agent
agent = ReflectionAgent(model="qwen/qwen3-32b")

# Run the agent with a user message
result = agent.run(
    user_msg="Write a Python function to calculate fibonacci numbers",
    n_steps=5,  # Number of reflection iterations
    verbose=1    # Enable detailed output
)

print(result)
```

### Key Features

- **Self-Critique**: The agent automatically evaluates and improves its own responses
- **Configurable Steps**: Control the number of reflection iterations
- **Custom Prompts**: Override default system prompts for generation and reflection
- **Verbose Logging**: Detailed output showing each generation and reflection step
- **Early Termination**: Stops when the agent determines no further improvements are needed

### Example Output

```
==================================================
STEP 1/5
==================================================

GENERATION

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

REFLECTION

The function works but has several issues:
1. No input validation
2. Inefficient recursive implementation
3. Missing docstring
4. No error handling

==================================================
STEP 2/5
==================================================

GENERATION

def fibonacci(n):
    """
    Calculate the nth Fibonacci number.
    
    Args:
        n (int): The position in the Fibonacci sequence
        
    Returns:
        int: The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 0:
        raise ValueError("Input must be non-negative")
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

REFLECTION

<OK>
```

### Configuration

You can customize the behavior by providing custom system prompts:

```python
custom_generation_prompt = """
You are an expert Python developer. Write clean, efficient, and well-documented code.
"""

custom_reflection_prompt = """
You are a code reviewer. Focus on:
- Code efficiency
- Error handling
- Documentation
- Best practices
"""

result = agent.run(
    user_msg="Create a REST API endpoint",
    generation_system_prompt=custom_generation_prompt,
    reflection_system_prompt=custom_reflection_prompt,
    n_steps=3
)
```

## Requirements

- Python 3.12+
- Groq API key (set in `.env` file)
- Required packages: `groq`, `colorama`, `python-dotenv`
