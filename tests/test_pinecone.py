from retrieval.vector_store import PineconeVectorStore


vector_store = PineconeVectorStore()

vector_store.create_index()

print("Pinecone index is ready.")
print("Index:", vector_store.index_name)
