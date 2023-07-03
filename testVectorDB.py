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
pinecone_env = r'northamerica-northeast1-gcp'
pinecone_api = r'37b5ca46-255b-460b-bd41-f4777ef56d3a'
# use open ai to do embeddings

openaikey = r'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'

model_name = r'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openaikey
)

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
processed_tables = _preprocess_tables(tables)
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







