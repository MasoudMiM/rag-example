import os
import logging
from datetime import datetime
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    connections,
)

INPUT_TEXT_FILE = './data/doc.txt' 
CREATE_VECTOR_DB = True
MILVUS_COLLEC_NAME = "gmesh_doc_collec"

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    log_filename = f'logs/log_gmsh_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    ) 

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def connect_to_milvus():
    connections.connect("default", host="localhost", port="19530")
    version = utility.get_server_version()
    logging.info(f"Milvus server version: {version}")
    logging.info("Connected to Milvus.")

def create_milvus_collection(collection_name):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)  
    ]
    schema = CollectionSchema(fields, description="Vector database for Gmsh documentation.")
    collection = Collection(name=collection_name, schema=schema)
    logging.info(f"Collection '{collection_name}' created in Milvus with fields {fields}.")
    return collection


def insert_vectors_to_milvus(collection, embeddings_list, original_texts):
    logging.info(f"Number of embeddings: {len(embeddings_list)}")
    
    data = [
        {"embedding": embedding, "text": original_text} 
        for embedding, original_text in zip(embeddings_list, original_texts)
    ]
    
    try:
        collection.insert(data) 
        logging.info(f"Inserted {len(embeddings_list)} vectors into Milvus.")
    except Exception as e:
        logging.error(f"Failed to insert data into Milvus: {e}")


def create_vector_database(text, chunk_size=5):
    logging.info("Creating vector database from extracted text.")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    
    sentences = sent_tokenize(text)
    logging.info(f"Number of sentences extracted: {len(sentences)}")
    
    text_chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    logging.info(f"Number of text chunks created: {len(text_chunks)}")
    
    logging.info(f"Sample text chunks: {text_chunks[:1]}")  # Log first chunk
    
    embeddings = model.encode(text_chunks)

    logging.info(f"Embeddings shape: {embeddings.shape}")

    embeddings_list = embeddings.tolist()

    connect_to_milvus()
    collection_name = MILVUS_COLLEC_NAME
    collection = create_milvus_collection(collection_name)

    insert_vectors_to_milvus(collection, embeddings_list, text_chunks) 

    index_params = {
        "index_type": "IVF_FLAT", 
        "metric_type": "L2",  
        "params": {"nlist": 100}
    }

    try:
        collection.create_index(field_name="embedding", index_params=index_params)
        logging.info(f"Index created for collection '{collection_name}'.")
    except Exception as e:
        logging.error(f"Failed to create index for collection '{collection_name}': {e}")

    try:
        collection.load() 
        logging.info(f"Collection '{collection_name}' loaded into memory.")
    except Exception as e:
        logging.error(f"Failed to load collection '{collection_name}': {e}")


if __name__ == "__main__":
    setup_logging()  
    text_content = read_text_file(INPUT_TEXT_FILE)
    
    create_vector_database(text_content)
    

