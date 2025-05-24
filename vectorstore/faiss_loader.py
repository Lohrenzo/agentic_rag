import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import Optional
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()


def load_vector_db():
    embeddings = OpenAIEmbeddings()

    print("About to load vector store/db")
    return FAISS.load_local(
        os.environ.get("AGENTIC_DB_DIR"),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def fetch_url_content(url: str) -> Optional[str]:
    """
    Fetches content from a URL by performing an HTTP GET request.

    Parameters:
        url (str): The endpoint or URL to fetch content from.

    Returns:
        Optional[str]: The content retrieved from the URL as a string,
                       or None if the request fails.
    """
    prefix_url: str = "https://r.jina.ai/"
    # Concatenate the prefix URL with the provided URL
    full_url: str = prefix_url + url

    try:
        response = requests.get(full_url)  # Perform a GET request
        if response.status_code == 200:
            return response.content.decode(
                "utf-8"
            )  # Return the content of the response as a string
        else:
            print(
                f"""
        Error: HTTP GET request failed with status code {response.status_code}
                """
            )
            return None
    except requests.RequestException as e:
        print(f"Error: Failed to fetch URL {full_url}. Exception: {e}")
        return None


def create_vector_db(
    url: Optional[
        str
    ] = """
    https://weaviate.io/blog/what-is-agentic-rag#:~:text=This%20section%20discusses%20the%20two,public%20information%20from%20web%20searches.
    """,
):
    """
    This function creates a vector database from the content of a given URL.
    It performs the following steps:
    1. Fetch content from the URL.
    2. Split content into chunks.
    3. Create embeddings.
    4. Persist vector database using Chroma/FAISS.

    Parameters:
        url: The URL to fetch content from. If None, a default URL is used.

    Returns:
        Optional[str]: The content retrieved from the URL as a string,
                       or None if the request fails.
    """
    if url is None or not url.strip():
        print("No URL provided. Please provide a valid URL.")
        return None

    print(f"Fetching content from URL: {url}")
    # # Fetch content from the URL
    content: Optional[str] = fetch_url_content(url)

    if content is not None:
        print("Content retrieved successfully:")
    else:
        print("Failed to retrieve content from the specified URL.")
        return None

    token_size = 150
    text_splitter = RecursiveCharacterTextSplitter(
        # model_name="gpt-4",
        chunk_size=token_size,
        chunk_overlap=0,
    )

    # Clean the content
    # content = clean_text(content)
    # print(f"Cleaned content length: {len(content)} characters")

    text_chunks = text_splitter.split_text(content)
    print(f"Total chunks: {len(text_chunks)}")
    print("First chunk:", text_chunks[0])
    print("First chunk length:", len(text_chunks[0]), "characters")

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Convert the chunks into Embeddings and store into FAISS Vector DB
    vector_db = FAISS.from_texts(text_chunks, embeddings)

    vector_db.save_local(os.environ.get("AGENTIC_DB_DIR"))
    print(
        "FAISS vectorstore saved locally at:",
        os.environ.get("AGENTIC_DB_DIR"),
    )
    print("Vector DB: ", vector_db.as_retriever())

    return vector_db.as_retriever()


if __name__ == "__main__":
    create_vector_db()
