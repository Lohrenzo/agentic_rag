from langchain.tools import tool
from langchain_tavily import TavilySearch


@tool
def web_search(search_query: str):
    """Useful for web searches"""
    # print("Web search tool has been called.\n")
    invoke = TavilySearch(max_results=2).invoke(f"{search_query}")
    result = ""
    if invoke["results"]:
        for x in invoke["results"]:
            result += f"\n{x['content']}"
    return result


@tool
def addition(a: float, b: float) -> str:
    """Useful for performing basic addition calculations with numbers"""
    # print("Addition tool has been called.\n")
    return f"The sum of {a} and {b} is {a + b}"


@tool
def subtraction(a: float, b: float) -> str:
    """Useful for performing basic subtraction calculations with numbers"""
    # print("Subtraction tool has been called.\n")
    return f"When {b} is subtracted from {a}, the answer is {a - b}"


@tool
def multiplication(a: float, b: float) -> str:
    """Useful for performing basic multiplication calculations with numbers"""
    # print("Multiplication tool has been called.\n")
    return f"{a} multiplied by {b} is {a * b}"


@tool
def division(a: float, b: float) -> str:
    """Useful for performing basic multiplication calculations with numbers"""
    # print("Division tool has been called.\n")
    return f"{a} divided by {b} is {a / b}"


@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user"""
    # print("Greeting tool has been called.\n")
    return f"Hello {name}, I hope you are well today"
