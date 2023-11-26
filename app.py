from fastapi import FastAPI
from gensim.models import KeyedVectors
import polars as pl

from processors.word2vec_processor import Word2VecProcessor
from datamodel import Phrase, PhraseResponse

app = FastAPI()

# Pre-fetch data needed
kv = KeyedVectors.load_word2vec_format(r'vectors/GoogleNews-vectors-negative300.bin', binary=True, limit=1000000)

phrases = pl.read_csv('data/inputs/phrases.csv', encoding='windows-1252')

phrase_list = list(phrases['Phrases'])

processor = Word2VecProcessor(keyed_vectors=kv, phrase_list=phrase_list)


@app.post("/")
async def return_hello() -> PhraseResponse:
    return {"Hello": "World"}


@app.post("/closest_phrase/")
async def closest_phrase(phrase: Phrase) -> PhraseResponse:
    closest_phrase = processor.closest_phrase(phrase.text)
    phrase_response = PhraseResponse(closest_phrase=closest_phrase)

    return phrase_response
