from pydantic import BaseModel


class Phrase(BaseModel):
    text: str


class PhraseResponse(BaseModel):
    closest_phrase: str
