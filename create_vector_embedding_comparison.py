import openai
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
import chromadb

#///////////////////////////////// LOAD CV
from pypdf import PdfReader
pdf_file=PdfReader('ResumeeZeynepTozge.pdf')
page = pdf_file.pages[0]



DEL = "text-embedding-ada-002"

os.environ["OPENAI_API_KEY"] = 'sk-proj-iDkWCfoha9vU1rH2g7UTT3BlbkFJAOd1X79ulRnnnBZgMABX'


# load the document and split it into chunks
loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embeddings = OpenAIEmbeddings()

# load it into Chroma
db = Chroma.from_documents(docs, embeddings )

embeddings = OpenAIEmbeddings()
new_client = chromadb.EphemeralClient()
openai_lc_client = Chroma.from_documents(
    docs, embeddings, client=new_client, collection_name="openai_collection"
)


docs = db.similarity_search_with_score()





























import plotly.express as px
from sklearn.manifold import TSNE

# Create a t-SNE model
tsne_model = TSNE(
    n_components = 2,
    perplexity = 15,
    random_state = 42,
    init = 'random',
    learning_rate = 200
)
tsne_embeddings = tsne_model.fit_transform(embedding_array)

# Create a DataFrame for visualisation
visualisation_data = pd.DataFrame(
    {'x': tsne_embeddings[:, 0],
     'y': tsne_embeddings[:, 1],
     'Similarity': df['normalised']}
)

# Create the scatter plot using Plotly Express
plot = px.scatter(
    visualisation_data,
    x = 'x',
    y = 'y',
    color = 'Similarity',
    color_continuous_scale = 'rainbow',
    opacity = 0.3,
    title = f"Similarity to '{query}' visualised using t-SNE"
)

plot.update_layout(
    width = 650,
    height = 650
)

# Show the plot
plot.show()