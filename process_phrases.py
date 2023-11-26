from itertools import combinations
import json

import numpy as np
from gensim.models import KeyedVectors
from gensim.utils import tokenize
import polars as pl

# Version 1 of this needs multiple upgrads:
# 1. Assign ids to each of the phrases
# 2. Use Gensim to get vectorised all phrases against all instead of python combinations


def phrase_to_vector(keyed_vectors: KeyedVectors, phrase: str) -> np.ndarray:
    vector = keyed_vectors.get_mean_vector(keys=list(phrase))

    return vector


def phrase_pair_to_distance(keyed_vectors: KeyedVectors, phrase_pair: (str, str)) -> float:
    # This function can have its efficiency improved by using Gensim to compare each to each other.
    # Open a story for this optimisation

    left_vector = phrase_to_vector(keyed_vectors, phrase_pair[0])

    right_vector = phrase_to_vector(keyed_vectors, phrase_pair[1])

    # Gensim cosine similarities compares 1 vector to many, so right side must be a matrix
    right_matrix = right_vector.reshape(1, len(right_vector))

    cosine_similarities = keyed_vectors.cosine_similarities(left_vector, right_matrix)

    return cosine_similarities[0].item()


def phrase_list_to_all_distances(keyed_vectors: KeyedVectors, phrase_list: list[str])\
        -> dict[tuple[str, str], np.float32]:

    phrase_vectors = {}

    for index, pair in enumerate(combinations(phrase_list, 2)):
        distance = phrase_pair_to_distance(keyed_vectors, pair)

        phrase_details = {}
        phrase_details['pair'] = pair
        phrase_details['distance'] = distance

        phrase_vectors[index] = phrase_details

    return phrase_vectors


if __name__ == "__main__":
    kv = KeyedVectors.load_word2vec_format(r'data/GoogleNews-vectors-negative300.bin', binary=True, limit=1000000)

    phrases = pl.read_csv('data/phrases.csv', encoding='windows-1252')

    phrase_vectors = phrase_list_to_all_distances(keyed_vectors=kv,
                                                  phrase_list=list(phrases['Phrases']))

    # The output format was not specified in requirements - output a json for now
    with open('data/phrase_vectors.json', 'w') as f:
        json.dump(phrase_vectors, f)

