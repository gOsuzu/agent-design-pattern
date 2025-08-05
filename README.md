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

## Tool Pattern

The Tool Pattern is an agent design pattern that enables AI agents to dynamically call external functions (tools) based on user requests. This pattern allows agents to extend their capabilities beyond their training data by using real-time data sources, APIs, and custom functions.

### How it Works

1. **Tool Registration**: Functions are decorated with `@tool` to make them available to the agent
2. **Signature Extraction**: The agent automatically extracts function signatures, descriptions, and type information
3. **Dynamic Tool Calling**: The agent decides when and which tools to call based on user requests
4. **Argument Validation**: Arguments are automatically validated and type-converted
5. **Result Integration**: Tool results are integrated into the agent's response

### Usage

#### Basic Tool Creation

```python
from agentic_patterns.tool_pattern.tool import tool
from agentic_patterns.tool_pattern.tool_agent import ToolAgent

# Create a simple tool
@tool
def get_weather(city: str) -> str:
    """Get current weather information for a city"""
    # Simulated weather data
    return f"Weather in {city}: Sunny, 25°C"

# Create a tool agent with the weather tool
agent = ToolAgent(tools=[get_weather])

# Use the agent
result = agent.run(user_msg="What's the weather like in Tokyo?")
print(result)
```

#### Advanced Tool with External API

```python
import requests
import json

@tool
def fetch_hacker_news_stories(top_n: int) -> str:
    """
    Fetch the top stories from Hacker News.
    
    Args:
        top_n (int): The number of top stories to retrieve
    """
    try:
        # Get top story IDs
        response = requests.get('https://hacker-news.firebaseio.com/v0/topstories.json')
        top_story_ids = response.json()[:top_n]
        
        stories = []
        for story_id in top_story_ids:
            story_url = f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json'
            story_response = requests.get(story_url)
            story_data = story_response.json()
            
            stories.append({
                'title': story_data.get('title', 'No title'),
                'url': story_data.get('url', 'No URL available'),
            })
        
        return json.dumps(stories)
    except Exception as e:
        return f"Error fetching stories: {e}"

# Create agent with multiple tools
agent = ToolAgent(tools=[get_weather, fetch_hacker_news_stories])

# The agent can now handle various requests
result1 = agent.run(user_msg="Get me the top 5 Hacker News stories")
result2 = agent.run(user_msg="What's the weather in London?")
```

#### Manual Tool Creation

```python
from agentic_patterns.tool_pattern.tool import tool, Tool

# Method 1: Using decorator (recommended)
@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two integers together"""
    return a + b

# Method 2: Manual creation
def multiply_numbers(x: int, y: int) -> int:
    """Multiply two integers"""
    return x * y

multiply_tool = tool(multiply_numbers)

# Both methods work the same way
agent = ToolAgent(tools=[calculate_sum, multiply_tool])
```

### Key Features

- **Automatic Signature Extraction**: Function names, descriptions, and type information are automatically extracted
- **Type Safety**: Arguments are validated and converted to the correct types
- **Smart Tool Selection**: The agent intelligently chooses which tools to use based on user requests
- **Error Handling**: Robust error handling for invalid tool calls and API failures
- **Flexible Integration**: Easy to add new tools without modifying the agent code
- **Natural Language Interface**: Users can request tool usage in natural language

### Tool Decorator Details

The `@tool` decorator automatically:

1. **Extracts function metadata**:
   - Function name
   - Docstring description
   - Parameter types and names

2. **Creates a Tool object** with:
   - `name`: Function name
   - `fn`: Original function
   - `fn_signature`: JSON-formatted signature for AI consumption

3. **Enables type validation**:
   - Converts string arguments to appropriate types
   - Validates required parameters
   - Handles type mismatches gracefully

### Example: Complete Workflow

```python
# 1. Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    return f"Search results for: {query}"

@tool
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle"""
    return length * width

# 2. Create agent
agent = ToolAgent(tools=[search_web, calculate_area])

# 3. Use the agent
# The agent will automatically choose appropriate tools
result1 = agent.run("What is the area of a rectangle with length 5 and width 3?")
result2 = agent.run("Search for information about Python programming")
result3 = agent.run("Hello, how are you?")  # No tools needed - direct response
```

### Error Handling

The Tool Pattern includes comprehensive error handling:

```python
# Invalid tool calls are handled gracefully
result = agent.run("Call a non-existent tool")
# Result: Error message explaining available tools

# Type conversion errors are handled
@tool
def process_number(n: int) -> str:
    return f"Processed: {n}"

# Even if the agent passes a string, it gets converted to int
result = agent.run("Process the number 42")
# Works correctly: "Processed: 42"
```

### Best Practices

1. **Clear Descriptions**: Provide detailed docstrings for your tools
2. **Type Annotations**: Always use type hints for better tool signature extraction
3. **Error Handling**: Include proper error handling in your tool functions
4. **Tool Granularity**: Create focused, single-purpose tools rather than complex multi-purpose ones
5. **Testing**: Test your tools independently before integrating with the agent

## Requirements

- Python 3.12+
- Groq API key (set in `.env` file)
- Required packages: `groq`, `colorama`, `python-dotenv`, `requests`
