# Agentic RAG Assistant

A Python-based AI assistant that combines retrieval-augmented generation (RAG) with OpenAI, a FAISS vector database, and a suite of tools (math, greetings, web search). The assistant first tries to answer questions using its local knowledge base; if it can't, it falls back to tools and web search.

---

## Features

- **Retrieval-Augmented Generation (RAG):** Answers questions using a local FAISS vector database of documents.
- **OpenAI LLM Integration:** Uses OpenAI’s GPT models for answer generation.
- **Tool Use:** Falls back to tools for web search, calculations, and greetings when the knowledge base is insufficient.
- **Seamless Routing:** Automatically chooses the best method to answer your question.

---

## Requirements

- Python 3.9+
- [OpenAI API key](https://platform.openai.com/)
- [Tavily API key](https://app.tavily.com/) (for web search)
- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

Install dependencies:

```sh
pip install -r requirements.txt
```

---

## Environment Variables

Set the following variables in a `.env` file in your project root:

OPENAI_API_KEY=your_openai_api_key

TAVILY_API_KEY=your_tavily_api_key

AGENTIC_DB_DIR=path/to/your/vectorstore

---

## Usage

1. **Prepare your vector database**  
   Use your `vectorstore/faiss_loader.py` (or similar) to ingest documents and create the FAISS vector store.
   You can rewite to use another vector DB like Chroma.

2. **Run the assistant**

   ```sh
   python main.py
   ```

3. **Interact**
   - Ask questions! The agent will first look in its knowledge base, then use tools if needed.
   - Type `quit` to exit.

---

## Example Interaction

Welcome! I'm your AI assistant. Type 'quit' to exit.
You can ask me to perform calculations, search the web, or answer from my knowledge base.

You: What is egusi soup made from?

Assistant (from knowledge base): Egusi soup is made from ground melon seeds, leafy vegetables, and assorted meats or fish. It is a popular West African dish.

You: What is the capital of France?

Assistant: The capital of France is Paris.

You: 15 + 27

Assistant: The sum of 15 and 27 is 42.

text

---

## Project Structure

.
├── main.py
├── utils/
│ └── tools.py
├── vectorstore/
│ └── faiss_loader.py
├── requirements.txt
├── .env
└── README.md

---

## Customization

- **Add More Tools:** Add functions to `utils/tools.py` with docstrings and decorate with `@tool`.
- **Expand Knowledge Base:** Ingest more documents into your FAISS vector store.
- **Change LLM Model:** Adjust the `model = ChatOpenAI(...)` line in `main.py`.

---

## Acknowledgments

- [LangChain](https://www.langchain.com/)
- [OpenAI](https://openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Tavily](https://www.tavily.com/)

---

**Enjoy your hybrid AI assistant!**  
Feel free to open issues or contribute.
