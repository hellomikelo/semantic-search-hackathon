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
    page_icon="🔍",
    # layout="wide",
    initial_sidebar_state="auto",
)

# Paste your API key here. Remember to not share publicly
api_key = os.getenv('COHERE_API_KEY')

# Create and retrieve a Cohere API key from os.cohere.ai
co = cohere.Client(api_key)

with st.sidebar:
    podcast_name = st.radio('Which podcast to search?', ('TFTS', 'McKinsey'))
    st.write('---\n')
    st.caption('''[Think Fast, Talk Smart](https://www.gsb.stanford.edu/business-podcasts/think-fast-talk-smart-podcast) (TFTS) is a podcast produced by Stanford Graduate School of Business. Each episode provides concrete, easy-to-implement tools and techniques to help you hone and enhance your communication skills.  
    [The McKinsey Podcast](https://www.mckinsey.com/featured-insights/mckinsey-podcast) (McKinsey) is a business and management podcast featuring conversations with leading experts. The series features topics ranging from leadership and change management to digital, analytics and deep-dives into specific regions and industries.
    ''')
podcast_name = podcast_name.lower()

df = pd.read_csv(f'./{podcast_name}_podcasts_embeds.csv', index_col=0, converters={'paragraphs': lambda x: x[2:-2].split("', '")})
texts = df.intro.to_numpy()

@st.cache(persist=True, allow_output_mutation=False, show_spinner=False, suppress_st_warning=True)
def hash_file_reference(file_reference):
    return f'./{podcast_name}_podcasts.ann'


@st.cache(persist=True, allow_output_mutation=False, show_spinner=False,
          suppress_st_warning=True, hash_funcs={annoy.AnnoyIndex: hash_file_reference})
def load_index(embed_dim=4096):
    # Load index
    search_index = AnnoyIndex(embed_dim, 'angular')
    search_index.load(f'./{podcast_name}_podcasts.ann')  # super fast, will just mmap the file
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

    # Explain reason for suggestion
    explanations = []
    for i in range(n_results):
        prompt_explain = f'How does this response relate to your request?\n\nYour request: "{query}"\n\nThis response: "{texts[similar_item_ids[0][i]]}"\n\nThis response relates to your request because'
        response = co.generate(
            model='command-xlarge-20221108',
            prompt=prompt_explain,
            max_tokens=100,
            temperature=0.8,
            k=0,
            p=1,
            frequency_penalty=0,
            presence_penalty=0,
            num_generations=1,
            # stop_sequences=["--"],
            return_likelihoods='NONE')
        explanations.append(response.generations[0].text.strip())

    # Format the results
    results_df = pd.DataFrame(data={
        'id': similar_item_ids[0],
        'text': texts[similar_item_ids[0]],
        'distance': similar_item_ids[1],
        'url': df.loc[similar_item_ids[0], 'url'],
        'title': df.loc[similar_item_ids[0], 'title'],
        'explanation': explanations
    })
    return results_df


@st.cache(persist=True, allow_output_mutation=False, show_spinner=False, suppress_st_warning=True)
def format_results(results_df):
    formatted_output = ''
    for i, row in results_df.reset_index().iterrows():
        # formatted_output += f'{i+1}. **[{row.title}]({row.url})** (score: {row.distance:0.4})\n{row.text}\n\n'
        formatted_output += f'{i + 1}. **[{row.title}]({row.url})**\n📝 {row.text}\n\n**🤔 Why this episode?** This episode is suggested because {row.explanation}\n\n'
    return formatted_output


# Streamlit app
st.title("🔍 Info Insighter")
query = st.text_input('What ideas would you like to listen to?', 'tips on how to accomplish more at work')
search_index = load_index()
results = get_search_results(query, search_index)
formatted_results = format_results(results)
st.write('---\nTop 5 recommendations:')
st.write(formatted_results)

# TODO: summarize episode chat
# TODO: follow up exercise questions