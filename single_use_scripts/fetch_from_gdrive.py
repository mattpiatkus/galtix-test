from gensim.models import KeyedVectors

# I got an error on the original
kv = KeyedVectors.load_word2vec_format(r'vectors/GoogleNews-vectors-negative300.bin', binary=True, limit=1000000)

kv.save_word2vec_format('data/vectors.csv')
