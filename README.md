# Simple_RAG

# The Problem
When working with a large language model to find information, the model is limited to information it learned when training which can lead to outdated information or the model hallucinating answers. A RAG (Retrieval Augmented Generation) system solves this problem by taking the user's query and matching it up against relevant information stored in a database. 

# Use instructions
This project was designed to run on a mac with apple's M-series chips. other devices may not work well or at all
1. Install the packages:
`pip install chromadb mlx_embeddings mlx_lm`
2. Download from [here](https://dumps.wikimedia.org/enwiki/latest/):
  - `enwiki-latest-pages-articles-multistream-index.txt.bz2`
  - `enwiki-latest-pages-articles-multistream.xml.bz2`
  - `enwiki-latest-redirect.sql`
3. Run the program:
`python rag-qa.py`
  - This will take a few hours to run the first time because it needs to vectorize all of the article titles and filter out the redirects for the system to work
4. Once you are done asking questions, type "END" in all caps to end the conversation


# How it works:
1. Vectorize article names if the vector database is empty
  - I used all-MiniLM-L6-v2 with 4-bit quantization as my embedding model because it runs well on my mac and because the article titles are only a few words, so TF-IDF would give very sparse vector representations that don't have enough information to be useful. The dense embeddings from this model gives me more information to work with.
  - I only vectorized the titles because the size of the fully decompressed article text is around 100 GB, and I don't have the storage or time to chunk and vectorize everything. I also couldn't just give the model all of the titles because there are around 7 million articles on Wikipedia and running them through every time would be wasteful.
2. Run in an infinite loop where every time a user asks a question:
    1. The model takes the question and uses it to come up with titles to potentially useful wikipedia articles
    2. These titles are vectorized, then being used to search for the most similar wikipedia document titles
    3. The metadata associated with these titles is used to index and find the corresponding articles
    4. These articles and the user's question are fed back into the model and it answers the question

