import base64
import os

import requests
import cohere
from annoy import AnnoyIndex
import annoy
import streamlit as st
# from fpdf import FPDF
from pydub import AudioSegment
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

st.set_page_config(
    page_title="Info Insighter",
    page_icon="alembic",
    # layout="wide",
    initial_sidebar_state="auto",
)

# Paste your API key here. Remember to not share publicly
api_key = os.getenv('COHERE_API_KEY')

# Create and retrieve a Cohere API key from os.cohere.ai
co = cohere.Client(api_key)

df = pd.read_csv('./mckinsey_podcasts_embeds.csv', index_col=0, converters={'paragraphs': lambda x: x[2:-2].split("', '")})
texts = df.intro.to_numpy()


def hash_file_reference(file_reference):
    return './podcasts.ann'


@st.cache(persist=True, allow_output_mutation=False, show_spinner=False,
          suppress_st_warning=True, hash_funcs={annoy.AnnoyIndex: hash_file_reference})
def load_index(embed_dim=4096):
    # Load index
    search_index = AnnoyIndex(embed_dim, 'angular')
    search_index.load('./podcasts.ann')  # super fast, will just mmap the file
    return search_index


@st.cache(persist=True, allow_output_mutation=False, show_spinner=False,
          suppress_st_warning=True, hash_funcs={annoy.AnnoyIndex: hash_file_reference})
def get_search_results(query, search_index, n_results=5):
    # Get the query's embedding
    query_embed = co.embed(texts=[query],
                           model="large",
                           truncate="LEFT").embeddings

    # Retrieve the nearest neighbors
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0], n_results,
                                                      include_distances=True)
    # Format the results
    results = pd.DataFrame(data={
        'id': similar_item_ids[0],
        'text': texts[similar_item_ids[0]],
        'distance': similar_item_ids[1],
        'url': df.loc[similar_item_ids[0], 'url'],
        'title' : df.loc[similar_item_ids[0], 'title']
    })
    return results


@st.cache(persist=True, allow_output_mutation=False, show_spinner=False, suppress_st_warning=True)
def format_results(df):
    formatted_output = ''
    for i, row in df.reset_index().iterrows():
        formatted_output += f'{i+1}. **[{row.title}]({row.url})**\n*{row.text}*\n\n'
    return formatted_output


# Streamlit app
st.title("Info Insighter")
query = st.text_input('What ideas are you interested in exploring?', 'tips on how to accomplish more in my work')
search_index = load_index()
results = get_search_results(query, search_index)
formatted_results = format_results(results)
st.write('---\nTop 5 recommendations from [The McKinsey podcast](https://www.mckinsey.com/featured-insights/mckinsey-podcast#):')
st.write(formatted_results)
