from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
from gensim.models import KeyedVectors


@dataclass
class Word2VecProcessor:
    """
    Holds word-to-vec keyed vectors and a phrase list and allows processing of phrases
    and comparison of new phrases against the list
    """
    keyed_vectors: KeyedVectors
    phrase_list: list

    indexed_phrases: Optional[dict[int, str]]  = field(default_factory=dict)
    phrase_vectors: Optional[np.ndarray] = np.ndarray
    all_phrase_distances: Optional[dict[int, dict[tuple[str, str], float]]] = field(default_factory=dict)

    def __post_init__(self):
        # Create a lookup of the phrases
        self.indexed_phrases = {index: phrase for index, phrase in enumerate(self.phrase_list)}

        # Vectorise all the phrases on class initiation.
        # TODO is this correct?  Might want to leave this till initiated by the user
        vectors = []
        for phrase in self.phrase_list:
            vectors.append(self.phrase_to_vector(phrase))

        self.phrase_vectors = np.vstack(vectors)

    def phrase_to_vector(self, phrase: str) -> np.ndarray:
        vector = self.keyed_vectors.get_mean_vector(keys=list(phrase))

        return vector

    def phrase_pair_to_distance(self, phrase_pair: (str, str)) -> float:
        # Instead of comparing phrases in pairs, efficiency could be increased by
        # using gensim and numpy to compare each phrase to each other (or each previous)

        left_vector = self.phrase_to_vector(phrase_pair[0])

        right_vector = self.phrase_to_vector(phrase_pair[1])

        # Gensim cosine similarities compares 1 vector to many, so right side must be a matrix
        right_matrix = right_vector.reshape(1, len(right_vector))

        cosine_similarities = self.keyed_vectors.cosine_similarities(left_vector, right_matrix)

        return cosine_similarities[0].item()

    def calculate_all_phrase_distances(self) -> None:
        """
        Calculates the distances between all phrases on the object
        Order matters - the first phrase appearing in the original data source will come first in the pair
        Therefore a lookup might need to reverse the pairs to find the distance
        Creates a new integer index for each pair
        TODO is this really the best data structure?
        :return: dict like {0, {'pair': ('I like cake',"cake is nice"), "distance": 0.99}}
        """
        phrase_distances = {}

        for index, pair in enumerate(combinations(phrase_list, 2)):
            distance = self.phrase_pair_to_distance(pair)

            phrase_details = {'pair': pair, 'distance': distance}

            phrase_distances[index] = phrase_details

        self.all_phrase_distances = phrase_distances

    def closest_phrase(self,
                       input_phrase: str) -> tuple[int, str]:
        """
        Computes the closest phrase in phrase_vectors to the input phrase
        :param phrase_vectors: numpy ndarray of phrase vectors
        :param input_phrase: str
        :return: The closest phrase to the input phrase
        """
        input_vector = self.phrase_to_vector(input_phrase)
        similarities = self.keyed_vectors.cosine_similarities(input_vector, self.phrase_vectors)
        top_similarity_index = np.argmax(similarities)

        return self.indexed_phrases[top_similarity_index]
