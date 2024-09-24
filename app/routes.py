from fastapi import APIRouter, UploadFile, File, Request
from .models import get_text_embedding, get_image_embedding

router = APIRouter()

@router.post("/embed_image/")
async def embed_image(file: UploadFile = File(...)):
    return {"embedding": await get_image_embedding(file)}


@router.post("/embed_text/")
async def embed_text(request: Request):
    body = await request.json()  # Parse the incoming JSON data
    text = body.get("text")  # Get the 'text' key from JSON
    if not text:
        return {"error": "Text parameter is missing."}
    print(f"Received text: {text}")
    return {"embedding": get_text_embedding(text)}
