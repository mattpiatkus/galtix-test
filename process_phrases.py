import json

from gensim.models import KeyedVectors
import polars as pl

from processors.word2vec_processor import Word2VecProcessor



if __name__ == "__main__":
    kv = KeyedVectors.load_word2vec_format(r'data/GoogleNews-vectors-negative300.bin', binary=True, limit=1000000)

    phrases = pl.read_csv('data/phrases.csv', encoding='windows-1252')

    phrase_list = list(phrases['Phrases'])

    processor = Word2VecProcessor(keyed_vectors=kv, phrase_list=phrase_list)

    processor.calculate_all_phrase_distances()

    all_phrase_distances = processor.all_phrase_distances

    # The output format was not specified in requirements - output a json for now
    with open('data/phrase_pair_vectors.json', 'w') as f:
        json.dump(all_phrase_distances, f)

