

## Goal

An app that can access your daily notes allowing you to ask questions about them.

Use Cases:
* Summarize the work that I completed this week?
* What updates should I provide for stand-up?

## Plan

### Index

Read from a directory of markdown or text files

Chunk the content from the files no larger than X

Add additional metadata like create/modified time and file name

Create embeddings for each chunk

Store embeddings in a simple local vector store (like ChromaDB)

### Query

created embeddings for the question

find the most similar chunks

sending the most similar chunks to a language model along with the query

get the response from the language model



