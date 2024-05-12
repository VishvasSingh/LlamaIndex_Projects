from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent


# DEFINING TOOLS
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""

    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)


llm = Ollama(model="llama3", request_timeout=120.0)

agent = ReActAgent.from_tools(
    tools=[multiply_tool, add_tool, subtract_tool], llm=llm, verbose=True
)

response = agent.chat("What is 40 + (100-30) * 5 ? Calculate step by step")

print(response)

prompt_dict = agent.get_prompts()
for k, v in prompt_dict.items():
    print(f"Prompt: {k} \n\nValue: {v.template}")
