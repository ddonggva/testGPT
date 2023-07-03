from datasets import load_dataset
from getpass import getpass
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import pandas as pd
import torch
from langchain.text_splitter import CharacterTextSplitter



# knowledge here
# https://docs.pinecone.io/docs/table-qa


# define macros
# pinecone credential
pinecone_env = r''
pinecone_api = r''
# use open ai to do embeddings

openaikey = r''

# model_name = r'text-embedding-ada-002'
#
# embed = OpenAIEmbeddings(
#     model=model_name,
#     openai_api_key=openaikey
# )

# load data
data = load_dataset("ashraq/ott-qa-20k", split="train")

# store all tables in the tables list
tables = []
# loop through the dataset and convert tabular data to pandas dataframes
for doc in data:
    table = pd.DataFrame(doc["data"], columns=doc["header"])
    tables.append(table)

print(tables[1])


# initialize retriever
# import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from sentence_transformers import SentenceTransformer

# load the table embedding model from huggingface models hub
retriever = SentenceTransformer("deepset/all-mpnet-base-v2-table", device=device)
retriever


# write table data into format
def _preprocess_tables(tables: list):
    processed = []
    # loop through all tables
    for table in tables:
        # convert the table to csv and
        processed_table = "\n".join([table.to_csv(index=False)])
        # add the processed table to processed list
        processed.append(processed_table)
    return processed

# format all the dataframes in the tables list
tabless = tables[:5]  # <--- original tables contains 20k tables, too large a throughput
processed_tables = _preprocess_tables(tabless)
# display the formatted table
processed_tables[2]


# create pinecone index
pinecone.init(
    api_key= pinecone_api,
    environment= pinecone_env
)
# you can choose any name for the index

index_name = "test-table-index-1"

# check if the table-qa index exists
if index_name not in pinecone.list_indexes():
    # create the index if it does not exist
    pinecone.create_index(
        index_name,
        dimension=768,
        metric="cosine"
    )

# connect to table-qa index we created
index = pinecone.Index(index_name)


# embeddings process
from tqdm.auto import tqdm

# we will use batches of 64
batch_size = 64

for i in tqdm(range(0, len(processed_tables), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(processed_tables))
    # extract batch
    batch = processed_tables[i:i_end]
    # generate embeddings for batch
    emb = retriever.encode(batch).tolist()
    # create unique IDs ranging from zero to the total number of tables in the dataset
    ids = [f"{idx}" for idx in range(i, i_end)]
    # add all to upsert list
    to_upsert = list(zip(ids, emb))
    # upsert/insert these records to pinecone
    _ = index.upsert(vectors=to_upsert)

# check that we have all vectors in index
index.describe_index_stats()

# test 1
query = "which Head Coach have the most NCAA Championships?"
query = "what is the total number of NCAA Championships by Notre Dame and North Carolina?"
# generate embedding for the query
xq = retriever.encode([query]).tolist()
# query pinecone index to find the table containing answer to the query
result = index.query(xq, top_k=1)
result


### read relevant table
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering

model_name = "google/tapas-base-finetuned-wtq"
# load the tokenizer and the model from huggingface model hub
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name, local_files_only=False)
# load the model and tokenizer into a question-answering pipeline
pipe = pipeline("table-question-answering",  model=model, tokenizer=tokenizer, device=device)

pipe(table=tables[3], query=query)


###################  replace pinecone with ChromaDB
from langchain.vectorstores import Chroma
text1=r'But we want to save the embeddings in a DB that is persistent because recreating them every time we open the application would be a resource waste. This is where ChromaDB helps us. We can create and save the embeddings using text parts, and add metadata for each part. In this, the metadata will be strings that name each text part.'

import chromadb
from chromadb.utils import embedding_functions

persist_directory = '/mnt/data/chromadb/'
chroma_client = chromadb.Client()

# collections are like tables in a db
collection1 = chroma_client.create_collection(name="collection1")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")



# describes how embeddings are done
# https://huggingface.co/blog/getting-started-with-embeddings


