import os
from typing import Any, List, Dict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as PineconeVectorStore

load_dotenv()

# Inicializar Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Crear el índice si no existe
if os.getenv("INDEX_NAME") not in pc.list_indexes().names():
    pc.create_index(
        name=os.getenv("INDEX_NAME"),
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=os.getenv("PINECONE_ENVIRONMENT")
        )
    )


def run_llm(query: str, chat_history: List[Dict[str, str]]) -> Any:
    # Crear objeto de embeddings de OpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Crear objeto de búsqueda basado en Pinecone
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=os.getenv("INDEX_NAME"), embedding=embeddings
    )

    # Crear modelo LLM de OpenAI
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Construir el historial del chat para incluirlo en el prompt
    history_text = ""
    for role, text in chat_history:
        if role == "human":
            history_text += f"Human: {text}\n"
        elif role == "ai":
            history_text += f"AI: {text}\n"

    # Crear el prompt completo con historial y la consulta actual
    full_prompt = f"{history_text}\nHuman: {query}"

    # Crear la cadena de preguntas y respuestas con la búsqueda
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )

    # Ejecutar la consulta con el historial y el prompt actual
    result = qa.invoke({"query": full_prompt})

    # Formatear el resultado final
    return {
        "query": query,
        "result": result["result"],  # Asumimos que el resultado está bajo esta clave
        "source": result["source_documents"]  # Si los documentos fuente son necesarios
    }