# imports
import pandas as pd
import pickle
from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer
import torch
from typing import Mapping
import requests
import numpy as np

import argparse
from datetime import datetime

from aslite.db import get_papers_db, load_users, save_embeddings, load_embeddings, EMBEDDINGS_FILE, safe_pickle_dump


# constants
model_id = "sentence-transformers/all-mpnet-base-v2"
# model_id = "sentence-transformers/all-MiniLM-L12-v2"
hf_token = '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'

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}
LOCAL = True
model = SentenceTransformer('all-mpnet-base-v2')


def load_cache(cache_fn):
    try:
        c = pd.read_pickle(cache_fn)
    except FileNotFoundError:
        c = {}
    return c


def get_strings(num_back, paper_database, key_dict: bool = False):
    """
    Return titles, abstracts, and authors from the database
    Parameters
    ----------
    num_back: int-- How many should be returned? -1 for all
    paper_database: paper database (pdb below)
    key_dict: bool=False, if True then return a dictionary with paper ids as keys
    Returns
    -------
    list of [[title, abstract, authors], [title2, abstract2, authors2], ... [titleN, abstractN, authorsN]]
    """

    # determine which papers we will use to build tfidf
    if num_back > -1 and num_back < len(paper_database):
        # crop to a random subset of papers
        paper_keys = list(paper_database.keys())
        z = [[x, datetime.strptime(pdb[x]['updated'], '%Y-%m-%dT%H:%M:%SZ')] for x in paper_keys]
        sorted_papers = sorted(z, key=lambda x: x[1])
        paper_keys = [x[0] for x in sorted_papers[-num_back:]]
    else:
        paper_keys = list(pdb.keys())

    data_back = []
    if key_dict:
        data_back = {}
    for p in paper_keys:
        d = paper_database[p]
        author_str = ' '.join([a['name'] for a in d['authors']])
        title = d["title"].replace('\n', ' ')
        abstract = d["summary"].replace('\n', ' ')
        total_string = f'Title: {title}. By: {author_str}. Abstract: {abstract}'
        # data_back.append([title, abstract, author_str])
        if key_dict:
            data_back[p] = total_string
        else:
            data_back.append(total_string)
    return data_back


def query_for_embedding(texts):
    if LOCAL:
        query_embedding = model.encode(texts)
        return query_embedding
    else:
        response = requests.post(api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
        return response.json()


def embedding_from_string(
        string: str,
        cache_fn: str = 'data/embeddings.pkl',
        cache_in: Mapping[str, list] = None,
) -> (list, Mapping):
    """Return embedding of given string, using a cache to avoid recomputing.
    Parameters:
        string: str -- string to be embedded
        cache_fn: str -- filename or path to the cache
        cache_in: Mapping[(str, str): vector] -- cache of existing embeddings.
    Returns:
        vector for the string under the model.
    """
    if cache_in is None or len(cache_in) == 0:
        cache_in = load_cache(cache_fn)
    cache_key = f'{string}'
    if cache_key not in cache_in.keys():
        cache_in[cache_key] = query_for_embedding(string)
        with open(cache_fn, "wb") as cache_f:
            pickle.dump(cache_in, cache_f)
    return cache_in[cache_key], cache_in


def embedding_from_strings(
        strings,
        cache_fn: str = 'data/embeddings.pkl',
        cache_in: Mapping[str, list] = None,
        key_dict: bool = False
) -> (list, Mapping):
    """Return embedding of given string, using a cache to avoid recomputing.
    Parameters:
        strings: list of strings OR dictionary of strings (if key_dict)
        cache_fn: str -- filename or path to the cache
        cache_in: Mapping[(str, str): vector] -- cache of existing embeddings.
        key_dict: bool -- is strings a dictionary?
    Returns:
        vector for the string under the model.
    """
    if cache_in is None or len(cache_in) == 0:
        cache_in = load_cache(cache_fn)
    if not key_dict:
        deleters = []
        for string in strings:
            if string in cache_in.keys():
                deleters.append(string)
        for string in deleters[::-1]:
            del strings[strings.index(string)]
    else:
        strings_to_embed = []
        embedding_keys = []
        for k in strings.keys():
            if k not in cache_in.keys():
                strings_to_embed.append(strings[k])
                embedding_keys.append(k)
        strings = strings_to_embed

    all_embeds_back = query_for_embedding(strings)

    for index in range(len(strings)):
        key_in = strings[index]
        if key_dict:
            key_in = embedding_keys[index]
        cache_in[key_in] = all_embeds_back[index]
    save_embeddings(cache_in)
    return all_embeds_back, cache_in


def print_recommendations_from_strings(
        comparison_text,
        comparison_embeddings,
        query_strings,
        query_embeddings,
        k_nearest_neighbors: int = 1,
):
    """Print out the k nearest neighbors of a given string."""
    # get embeddings for all strings
    hits = semantic_search(torch.tensor(query_embeddings), torch.tensor(comparison_embeddings),
                           top_k=k_nearest_neighbors)
    # print out source string
    # print out its k nearest neighbors
    for string, hit in zip(query_strings, hits):
        print(f"Source string: {string}")
        # print out the similar strings and their distances
        for i in range(len(hit)):
            print(
                f"""
            --- Recommendation #{i} (nearest neighbor {i} of {k_nearest_neighbors}) ---
            String: {comparison_text[hit[i]['corpus_id']]}
            Distance: {hit[i]['score']:0.3f}"""
            )

    return hits


def get_recommendations_embed(
        comparison_embeddings,
        query_embeddings,
):
    hits = semantic_search(torch.from_numpy(query_embeddings), torch.from_numpy(np.array(comparison_embeddings)),
                           top_k=len(comparison_embeddings))
    return hits[0]


def init_user():
    return np.random.randn(768, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arxiv Computor')
    parser.add_argument('--max_docs', type=int, default=-1,
                        help='maximum number of documents to use, or -1 to disable')
    parser.add_argument('--test', '-t', action="store_true", help="Test embedding functions?")
    args = parser.parse_args()
    print(args)
    if args.test:
        users = load_users()
        test_user = list(users.keys())[0]
        test_user = users[test_user]
        papers = [v for k, v in load_embeddings().items()]
        get_recommendations_embed(papers, test_user)

    else:
        # load the cache if it exists, and save a copy to disk
        embedding_cache = load_embeddings()

        pdb = get_papers_db(flag='r')

        strs = get_strings(num_back=-1, paper_database=pdb, key_dict=True)

        all_embeddings, embedding_cache = embedding_from_strings(strs,
                                                                 cache_fn=EMBEDDINGS_FILE,
                                                                 cache_in=embedding_cache,
                                                                 key_dict=True)
        save_embeddings(embedding_cache)

        pids = []
        feats = []
        for key, value in embedding_cache.items():
            pids.append(key)
            feats.append(value)
        features = {
            'pids': pids,
            'x': feats,
        }

        feature_file_path = 'data/sentence_features.pt'
        safe_pickle_dump(features, feature_file_path)
