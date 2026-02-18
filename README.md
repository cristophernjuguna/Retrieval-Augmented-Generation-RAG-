# Retrieval-Augmented-Generation-RAG-
Policy Document RAG System Using ChromaDB
Finance Bill 2025
Objective

This project implements a Retrieval-Augmented Generation (RAG) retrieval system using the Finance Bill 2025 as the policy document. The system extracts text from the PDF, splits it into chunks, generates embeddings using a transformer model, stores the embeddings in ChromaDB, and retrieves the most relevant sections in response to a user query.

Step 1 — Creating a Persistent ChromaDB Client
import chromadb

client = chromadb.PersistentClient(path='./chromadb')

Explanation

chromadb.PersistentClient creates a local vector database that stores data on disk inside the ./chromadb directory. This means the database persists between notebook runs.

If chromadb.Client() were used instead, the database would exist only in memory and would be erased once the session ends.

Persistence ensures that previously stored embeddings and documents remain available for future queries.

Step 2 — Creating or Retrieving a Collection
collection = client.get_or_create_collection(
    name='finance_collections',
    metadata={'description': 'my_finance_collection'}
)

print('collection_created:', collection.name)

Explanation

A collection in ChromaDB is similar to a table in a traditional database. It stores:

Document text

Embeddings

Unique identifiers (IDs)

Optional metadata

get_or_create_collection checks whether the collection exists.
If it does, it retrieves it.
If it does not, it creates it.

Collection-level metadata is optional information describing the collection itself.

Step 3 — Preparing Unique IDs
ids = [f"doc_{i}" for i in range(len(chunks))]

Explanation

Each document chunk must have a unique identifier.

This line generates IDs such as:

doc_0
doc_1
doc_2
...

The number of IDs must match the number of chunks and embeddings.

Step 4 — Adding Documents and Embeddings to the Collection
collection.add(
    documents=chunks,
    embeddings=finance_embeddings,
    ids=ids
)

Explanation

This stores the data inside ChromaDB.

documents → the text chunks extracted from the Finance Bill

embeddings → numerical vector representations of each chunk

ids → unique identifiers for each chunk

The embeddings and documents must be in the same order.

ChromaDB builds a vector index internally, allowing fast similarity search later.

Step 5 — Counting Stored Records
collection.count()

Explanation

Returns the number of records currently stored in the collection.

If the number is higher than expected, it may indicate that the collection already contained data from previous runs because the database is persistent.

Step 6 — Querying the Collection
Defining a User Query
query = 'what does the finance bill say about taxes'


This represents a natural language question from a user.

Generating Query Embedding
query_embedding = model.encode([query]).tolist()


The query is converted into a vector using the same embedding model used for the documents.

Using the same model and preprocessing ensures that similarity comparisons are valid.

Performing Vector Search
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

results

Explanation

collection.query() performs similarity search using the query embedding.

Parameters:

query_embeddings → the embedded user question

n_results=3 → returns the top 3 most similar document chunks

The returned result includes:

ids → IDs of matching chunks

documents → actual text of retrieved chunks

distances → similarity/distance scores

Understanding Similarity Scores

ChromaDB typically returns cosine distance.

Cosine distance = 1 − cosine similarity.

Lower distance means the vectors are closer in semantic space.

If vectors are normalized:

Smaller distance → stronger semantic match

Larger distance → weaker match

For example, distances around 0.48–0.50 indicate moderate similarity.

Key Concepts
Embedding

A numerical vector representation of text that captures semantic meaning.
Text with similar meaning will produce vectors that are close together in high-dimensional space.

Vector Database

A database optimized for storing and searching embeddings using similarity metrics.

Cosine Similarity

A measure of how aligned two vectors are. It evaluates similarity based on direction rather than magnitude.

Metadata

Additional structured information stored with each document (e.g., source file name, page number). Metadata enables traceability and document referencing.

Chunking

The process of dividing a large document into smaller segments before embedding. This improves retrieval precision.

Persistence

Storing database files on disk so data remains available after restarting the environment.

