# import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from vectorstore.faiss_loader import load_vector_db  # Your existing loader
from langchain.prompts import PromptTemplate
import re

from utils.tools import (
    addition,
    subtraction,
    multiplication,
    division,
    say_hello,
    web_search,
)

load_dotenv()


# --- Vector DB QA Function ---
def vector_db_answer(question, model):
    vector_db = load_vector_db()
    docs = vector_db.similarity_search(question)
    # Filter out junk docs
    filtered_docs = [
        doc
        for doc in docs
        if len(doc.page_content.strip()) > 20
        and re.search(r"[a-zA-Z]", doc.page_content)
    ]
    if not filtered_docs:
        return None
    context = "\n".join([doc.page_content for doc in filtered_docs])
    # Prompt for concise answer
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        Use three sentences maximum and keep the answer concise.
        If you don't know the answer, just say you don't know.
        If the answer is not in the context, just say you don't know.

        Context:
        {context}

        Question:
        {question}
        """,
    )
    chain = prompt | model
    result = chain.invoke({"context": context, "question": question})
    # If the answer is "I don't know" or similar, treat as not found
    if "don't know" in result.content.lower():
        return None
    return result.content


# Main Agent Loop
def main():
    model = ChatOpenAI(temperature=0)
    tools = [
        addition,
        subtraction,
        multiplication,
        division,
        say_hello,
        web_search,
    ]
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print(
        """You can ask me to perform calculations, search the web,
        or answer from my knowledge base."""
    )

    while True:
        user_input = input("\nYou: ").strip()
        if user_input == "quit":
            break

        # 1. Try to answer from vector DB
        answer = vector_db_answer(user_input, model)
        if answer:
            # print("\nAssistant (from knowledge base):", answer)
            print("\nAssistant:", answer)
            continue

        # 2. Fallback to agent tools (web search, math, etc.)
        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()


if __name__ == "__main__":
    main()
