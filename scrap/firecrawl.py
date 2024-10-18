import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from firecrawl.firecrawl import FirecrawlApp

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Función para cargar datos desde una URL usando Firecrawl
def load_document_from_url(url: str) -> Document:
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    page_content = app.scrape_url(url=url, params={"onlyMainContent": True})

    document = Document(page_content=str(page_content), metadata={"source": url})
    return document

# Función para procesar y subir el contenido a Pinecone
def ingest_firecrawl_data(url: str) -> None:
    document = load_document_from_url(url)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    documents = text_splitter.split_documents([document])

    print(f"Going to add {len(documents)} chunks from {url} to Pinecone")

    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=os.getenv("INDEX_NAME")
    )

    print(f"Successfully inserted {len(documents)} chunks into Pinecone")

if __name__ == "__main__":
    url_to_scrape = "https://example.com"
    ingest_firecrawl_data(url_to_scrape)