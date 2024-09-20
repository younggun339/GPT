from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Ilario Cannavaro",
    description="Get a real quote said by Ilario Cannavaro himself.",
    servers=[{"url": "https://candidate-compact-streaming-jackie.trycloudflare.com"}]
)


class Quote(BaseModel):
    quote: str = Field(
        description="The quote that Ilario Cannavaro said.",
    )
    year: int = Field(
        description="The year when Ilario Cannavaro said the quote.",
    )


@app.get(
    "/quote",
    summary="Returns a random quote by Ilario Cannavaro",
    description="Upon receiving a GET request this endpoint will return a real quiote said by Ilario Cannavaro himself.",
    response_description="A Quote object that contains the quote said by Ilario Cannavaro and the date when the quote was said.",
    response_model=Quote,
)
def get_quote():
    return {
        "quote": "역시 사이좋게 지내야한다고 생각해.",
        "year": 1950,
    }