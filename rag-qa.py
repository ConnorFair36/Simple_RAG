# Author: Connor Fair
# Class:  CMSC 437 Intro to Natural Language Processing
# Date:   3-12-26

# The Problem
#  When working with a large language model to find information, the model is limited to information it learned when training
#   which can lead to outdated information or the model halucinating answers.
#  A RAG (Retreval Augmented Generation) system solves this problem by taking the user's query and matching it up against relevent 
#   information stored in a database.

# Use instructions
#  This project was designed to run on a mac with apple's M-series chips. other devices may not work well or at all
#  1. Install the packages:
#    - pip install chromadb mlx_embeddings mlx_lm
#  2. Download the most recent wiki dump and index file (use the multi-stream version)
#  3. Run the program:
#    - python rag-qa.py
#    - This will take a few hours to run the first time because it needs to vectorize all of the article titles for the system to work
#  4. Once you are done asking questions, type "END" in all caps to end the conversation


# How it works:
# 1. Vectorize article names if the vector database is empty
#   - I used all-MiniLM-L6-v2 with 4-bit quantization as my embedding model because it runs well on my mac and because the article 
#      titles are only a few words, so TF-IDF would give very sparse vector reperesentations that don't have enough information to
#      be useful. The dense embeddings from this model gives me more information to work with.
#   - I only vectorized the titles because the size of the fully decompressed article text is around 100 GB, and I don't have the
#      storage or time to chunk and vectorize everything. I also couldn't just give the model all of the titles because there are 
#      around 7 million articles on Wikipedia and running them through everytime would be wasteful.
# 2. Run in an infinite loop where everytime a user asks a question:
#  1. The model takes the question and uses it to come up with titles to potentaly useful wikipedia articles
#  2. These titles are vectorized, then being used to search for the most similar wikipedia document titles
#  3. The metadata associated with these titles is used to index and find the corresponding articles
#  4. These articles and the user's question are fed back into the model and it awnsers the question


# reading wikidumps
import bz2
import os
import xml.sax
import io
import re

# storing and accessing vector databases
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction, register_embedding_function
import uuid
from mlx_embeddings.utils import load as embed_load
import numpy as np

# loading gemma 4 on mac
from mlx_lm import load, generate

# fancy output :)
from rich.console import Console
from rich.markdown import Markdown

@register_embedding_function
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name):
        self.model, self.tokenizer = embed_load(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        inputs =  self.tokenizer.batch_encode_plus(input, padding=True, return_tensors="mlx")
        outputs = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        batch_embeddings = outputs.text_embeds
        return np.array(batch_embeddings)

    @staticmethod
    def name() -> str:
        return "my-ef"

    def get_config(self):
        return dict(model=self.model)

    @staticmethod
    def build_from_config(config):
        return MyEmbeddingFunction(config['model'])


class SingleArticleExtractor(xml.sax.ContentHandler):
    def __init__(self, target_id=None, target_title=None):
        super().__init__()
        self.target_id = target_id
        self.target_title = target_title
        self.found_article = None

        self.in_page = False
        self.read_stack = []
        self.text_aggregate = []
        self.page_id = None
        self.page_title = None
        self.page_ns = None
        self.page_redirect = None
        self.page_content = None

    def startElement(self, tag_name, attributes):
        self.text_aggregate = []
        if tag_name == "page":
            self.in_page = True
            self.page_id = self.page_title = self.page_ns = None
            self.page_redirect = self.page_content = None
        elif tag_name == "redirect" and self.read_stack and self.read_stack[-1] == "page":
            self.page_redirect = attributes["title"]
        self.read_stack.append(tag_name)

    def endElement(self, tag_name):
        if self.read_stack and tag_name == self.read_stack[-1]:
            del self.read_stack[-1]

        element_string = ''.join(self.text_aggregate)

        if tag_name == "page":
            self.in_page = False
            # Check if this is the article we want
            if (self.target_id and self.page_id == self.target_id) or \
               (self.target_title and self.page_title == self.target_title):
                self.found_article = {
                    "id":       self.page_id,
                    "title":    self.page_title,
                    "ns":       self.page_ns,
                    "redirect": self.page_redirect,
                    "content":  self.page_content,
                }
        elif self.in_page:
            if self.read_stack and self.read_stack[-1] == "page":
                if tag_name == "title":
                    self.page_title = element_string
                elif tag_name == "id":
                    self.page_id = int(element_string)
                elif tag_name == "ns":
                    self.page_ns = int(element_string)
            elif self.read_stack and self.read_stack[-1] == "revision":
                if tag_name == "text":
                    self.page_content = element_string

    def characters(self, content):
        if self.in_page:
            self.text_aggregate.append(content)


def vectorize_titles(collection, index_source):
    """Pulls all of the titles out of the index file and vectorizes each one."""
    batch_size = 5_000
    total_batches = 0
    
    # Resume from where we left off
    skip_lines = collection.count()
    print(f"Skipping {skip_lines} already-vectorized titles...")

    batch = {
        "ids": [],
        "documents": [],
        "metadatas": []
    }
    try:
        index_filehandle = bz2.BZ2File(index_source)
        for i, line in enumerate(index_filehandle):
            if i < skip_lines:
                continue  # skip already-processed lines
            
            data = line.decode("utf-8").split(":")
            line_index = str(uuid.uuid4())
            batch["ids"].append(line_index)
            batch["documents"].append(data[2])
            batch["metadatas"].append({"offset": data[0], "page_id": data[1]})

            if len(batch["ids"]) >= batch_size:
                collection.add(
                    ids=batch["ids"],
                    documents=batch["documents"],
                    metadatas=batch["metadatas"],
                )
                batch = {"ids": [], "documents": [], "metadatas": []}
                total_batches += 1
                print(f"Batch {total_batches} finished.")
                print(f"Collection size: {collection.count()}")

        # Don't forget the final partial batch
        if batch["ids"]:
            collection.add(
                ids=batch["ids"],
                documents=batch["documents"],
                metadatas=batch["metadatas"],
            )
            print(f"Final batch finished. Collection size: {collection.count()}")
    finally:
        index_filehandle.close()


def get_article(metadatas, article_source):
    """Gets an article out of the wikipedia dump based on the metadata."""
    offset = int(metadatas["offset"])
    target_id = int(metadatas["page_id"])
    with open(article_source, "rb") as f:
        f.seek(offset)                          # jump straight to byte offest to skip lots of file reading
        decompressor = bz2.BZ2Decompressor()

        output = [b'<pages>']
        while not decompressor.eof:
            output.append(decompressor.decompress(f.read(65536)))
        output.append(b'</pages>')

    contents = b''.join(output)

    # Now parse the XML fragment and find the specific article
    handler = SingleArticleExtractor(target_id=target_id, target_title=None)
    xml.sax.parse(io.BytesIO(contents), handler)
    return handler.found_article


def filter_article(article):
    """Removes text that adds no context from the document."""
    # remove references
    article = article.split("== References")[0]
    article = re.sub(r'(?:<ref.*/>)', '', article)
    article = re.sub(r'(?:<ref>.*</ref>)', '', article)
    return article


def get_relevent_docs(collection, article_source, llm_rag_query, n_documents):
    """Gets the n most relevent documents out the wikipedia dump."""
    query = collection.query(
        query_texts=llm_rag_query,
        n_results=n_documents
    )
    metadatas = query["metadatas"][0]
    documents = []
    for metadata in metadatas:
        documents.append(get_article(metadata, article_source))
    redirects = [doc["content"] for doc in documents if doc["redirect"]]
    # find the articles for the redirects in the database if possible
    if redirects:
        query = collection.query(
            query_texts=llm_rag_query,
            n_results=len(redirects)
        )
        metadatas = query["metadatas"][0]
        new_documents = []
        for metadata in metadatas:
            new_documents.append(get_article(metadata, article_source))
        documents.extend(new_documents)
    # remove as many redirects as possible
    documents = [filter_article(doc["content"]) for doc in documents if not doc["redirect"]]

    return documents[:n_documents]


def main(collection, article_source, n_titles: int, n_documents: int):
    console = Console()
    # Load 4-bit quantized version for speed
    model, tokenizer = load("mlx-community/gemma-4-e2b-it-4bit")

    while True:
        # Get user question
        new_message = console.input("> ")
        if new_message == "END":
            console.print("Bye!")
            break
        # Convert question into useful article titles for vector search
        rag_prompt = f"""Using the user question attached to this message, come up with {n_titles} possible wikipedia article titles that could contain useful information to help answer the question: {new_message}
This response must be in the format: title 1|title 2|title 3... with nothing else in the response."""
        messages = [{"role": "user", "content": rag_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        llm_rag_query = generate(model,
                         tokenizer,
                         prompt=prompt,
                         #verbose=True,
                         max_tokens=100000)
        
        llm_rag_query = llm_rag_query.split("<channel|>")[-1].split("|")
        console.print(llm_rag_query)

        documents = get_relevent_docs(collection, article_source, llm_rag_query, n_documents)
        console.print(documents)



        # Combine the articles found with the user's query to answer the question
        messages = [{"role": "user", "content": f"Context: {documents}\nQuestion: {new_message}"}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        output = generate(model,
                         tokenizer,
                         prompt=prompt,
                         #verbose=True,
                         max_tokens=100000)

        output = output.split("<channel|>")

        console.print(Markdown("# Thinking"))
        console.print(Markdown(output[0]))
        console.print(Markdown("# Response"))
        console.print(Markdown(output[1]))


if __name__ == "__main__":
    # THE PLAN!
    # 1. Download all of the articles from wikipedia in the dump
    index_source = "./enwiki-latest-pages-articles-multistream-index.txt.bz2"
    article_source = "./enwiki-latest-pages-articles-multistream.xml.bz2"

    # 2. Vectorize article names
    my_ef = MyEmbeddingFunction("mlx-community/all-MiniLM-L6-v2-4bit")
    client = chromadb.PersistentClient(path="./wiki-vectors")
    collection = client.get_or_create_collection(
                name="my_collection", 
                embedding_function=my_ef
            )

    if collection.count() == 0:
        print("Collection is empty")
        vectorize_titles(collection, index_source)
    
    # 3. Run in an infinite loop where everytime a user asks a question:
    #  1. The model takes the question and uses it to come up with titles to potentaly useful wikipedia articles
    #  2. These titles are vectorized, then being used to search for the most similar wikipedia document titles
    #  3. The metadata associated with these titles is used to index and find the corresponding articles
    #  4. These articles and the user's question are fed back into the model to improve the response of the model
    n_titles = 3
    main(collection, article_source, n_titles, 3)

