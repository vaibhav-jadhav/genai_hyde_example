from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# Load environment variables
load_dotenv()

# Configuration
PDF_PATH = "python.pdf"
COLLECTION_NAME = "learning_langchain-hyde"
QDRANT_URL = "http://localhost:6333"
THRESHOLD = 0.50
QUERY = "How to loop over two lists together?"

# Initialize clients
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(url=QDRANT_URL)
openai_client = OpenAI()
system_prompt = """
You are an AI Assistant who can take users' Python documentation queries and answer how to perform operations in Python.
The user might be a newbie and may ask vague or worded-differently queries. Answer them properly.
Your answers must be short and correct.
"""


def load_and_split_pdf(pdf_path: str):
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    # Optional splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    return docs  # or splitter.split_documents(docs)


def inject_into_qdrant(split_docs):
    print("⚙️ Injecting documents into Qdrant...")
    vector_store = QdrantVectorStore.from_documents(
        documents=[],
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embedding=embedder,
        force_recreate=True
    )
    vector_store.add_documents(documents=split_docs)
    print("✅ Injection done!")


def search_with_threshold(query: str):
    retriever = QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embedding=embedder
    )

    results = retriever.similarity_search_with_score(query, k=3)
    filtered = [(doc, score) for doc, score in results if score >= THRESHOLD]

    if filtered:
        print("\n✅ Relevant Chunks Found:\n")
        for doc, score in filtered:
            print(f"Score: {score:.2f}")
            print("Chunk:", doc.page_content)
            print("-" * 50)
    else:
        print("\n❌ No relevant data found for this query.")


def generate_hypothetical_answer(query: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": query }
        ]
    )
    answer = response.choices[0].message.content
    print("\n🤖 Hypothetical Answer (HyDE):\n", answer)
    return answer


def main():
    # Step 1: Load & Inject PDF
    docs = load_and_split_pdf(PDF_PATH)
    inject_into_qdrant(docs)

    # Step 2: Search without HyDE
    print("\n🔍 Searching without HyDE:")
    search_with_threshold(QUERY)

    # Step 3: Generate Hypothetical Answer via LLM
    hypo_query = generate_hypothetical_answer(QUERY)

    # Step 4: Search again with HyDE-enhanced query
    print("\n🔍 Searching with HyDE-enhanced query:")
    search_with_threshold(hypo_query)


if __name__ == "__main__":
    main()
