"""
Extracts tfidf features from all paper abstracts and saves them to disk.
"""

import argparse
from random import shuffle
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from aslite.db import get_papers_db, save_features


# -----------------------------------------------------------------------------
def get_strings(num_back, paper_database):
    """
    Return titles, abstracts, and authors from the database
    Parameters
    ----------
    num_back: int-- How many should be returned? -1 for all
    paper_database: paper database (pdb below)
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

    # yield the abstracts of the papers
    data_back = []
    for p in paper_keys:
        d = paper_database[p]
        author_str = ' '.join([a['name'] for a in d['authors']])
        total_string = f'{d["title"]}. By: {author_str}. Abstract: {d["summary"]}'
        # data_back.append([d['title'], d['summary'], author_str])
        data_back.append(total_string)
    return data_back


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arxiv Computor')
    parser.add_argument('-n', '--num', type=int, default=20000, help='number of tfidf features')
    parser.add_argument('--min_df', type=int, default=5, help='min df')
    parser.add_argument('--max_df', type=float, default=0.1, help='max df')
    parser.add_argument('--max_docs', type=int, default=-1,
                        help='maximum number of documents to use when training tfidf, or -1 to disable')
    args = parser.parse_args()
    print(args)

    v = TfidfVectorizer(input='content',
                        encoding='utf-8', decode_error='replace', strip_accents='unicode',
                        lowercase=True, analyzer='word', stop_words='english',
                        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
                        ngram_range=(1, 2), max_features=args.num,
                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                        max_df=args.max_df, min_df=args.min_df)

    pdb = get_papers_db(flag='r')

    strs = get_strings(10, pdb)

    def make_corpus(training: bool):
        assert isinstance(training, bool)

        # determine which papers we will use to build tfidf
        if training and args.max_docs > 0 and args.max_docs < len(pdb):
            # crop to a random subset of papers
            keys = list(pdb.keys())
            shuffle(keys)
            keys = keys[:args.max_docs]
        else:
            keys = pdb.keys()

        # yield the abstracts of the papers
        for p in keys:
            d = pdb[p]
            author_str = ' '.join([a['name'] for a in d['authors']])
            yield ' '.join([d['title'], d['summary'], author_str])


    print("training tfidf vectors...")
    v.fit(make_corpus(training=True))

    print("running inference...")
    x = v.transform(make_corpus(training=False)).astype(np.float32)
    print(x.shape)

    print("saving to features to disk...")
    features = {
        'pids': list(pdb.keys()),
        'x': x,
        'vocab': v.vocabulary_,
        'idf': v._tfidf.idf_,
    }
    save_features(features)
