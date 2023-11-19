from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()
chroma_db_path = os.getenv('CHROMA_DB_PATH')

chroma_client = chromadb.PersistentClient(path=chroma_db_path)

def add_record(document, id, db_alias):
    collection_name = db_alias
    target_collection = chroma_client.get_or_create_collection(collection_name)
    target_collection.upsert(documents=document, metadatas=[{'type': 'question'}], ids=id)

def query_record(query_texts, num_results, db_alias):
    collection_name = db_alias
    target_collection = chroma_client.get_collection(collection_name)
    query_results = target_collection.query(query_texts=query_texts, n_results=num_results)
    return convert_to_pinecone_object_model(query_results)

def delete_record(db_alias, id):
    collection_name = db_alias
    target_collection = chroma_client.get_collection(collection_name)
    target_collection.delete(ids=[id])

def get_all_records(db_alias: str, nCount: int):
    collection_name = db_alias
    target_collection = chroma_client.get_collection(collection_name)
    return target_collection.peek(limit=nCount)

def convert_to_pinecone_object_model(chroma_results: dict):
    results = []
    for i in range(len(chroma_results["ids"][0])):
        results.append(
            {
                "question": chroma_results["documents"][0][i],
                "sql_query": chroma_results["ids"][0][i],
                "distance": chroma_results["distances"][0][i],
            }
        )
    return results