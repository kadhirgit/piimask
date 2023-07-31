import re

from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import spacy
# import spacy_transformers
# import tensorflow
from typing import List

app = FastAPI()

# Load the spaCy NER model
nlp = spacy.load("en_core_web_lg")

# Custom spaCy pipeline component to recognize credit card numbers
@spacy.Language.component("credit_card_component")
def credit_card_component(doc):
    for token in doc:
        if re.match(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', token.text):
            doc.ents += (spacy.tokens.Span(doc, token.i, token.i + 1, label="CARD"),)
    return doc

# Add the custom pipeline component to the spaCy model
# Add the custom pipeline component to the spaCy model
if "credit_card_component" not in nlp.pipe_names:
    nlp.add_pipe("credit_card_component",before="ner")

# Pydantic model for input text
class InputText(BaseModel):
    text: str
    blacklist: str

# Pydantic model for NER entity
class Entity(BaseModel):
    text: str
    start_pos: int
    end_pos: int
    label: str

# Function to mask specific patterns in the input text
def mask_patterns(input_text: str, entities: List[Entity]):

    
    # Mask email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    input_text = re.sub(email_pattern, "*EMAIL*", input_text)

    # Mask URLs
    url_pattern = r'https?://\S+|www\.\S+'
    input_text = re.sub(url_pattern, "*URL*", input_text)

    # Mask credit card numbers (Simple pattern matching for example purposes only)
    credit_card_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    input_text = re.sub(credit_card_pattern, "*CARD*", input_text)

    # Mask phone numbers (Simple pattern matching for example purposes only)
    phone_number_pattern = r'\+?\d{1,2}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    input_text = re.sub(phone_number_pattern, "*PHONENO*", input_text)


        # Sort entities in reverse order by end position to avoid conflicts while inserting the mask
    #entities = sorted(entities, key=lambda x: x.end_pos, reverse=True)

    # Mask NER entities
    # for entity in entities:
    #   if (entity.label !="CARDINAL") and (entity.text.lower() not in '*CARD*'):  
    #     mask = f"*<{entity.label}>*"
    #     length_diff = len(mask) - (entity.end_pos - entity.start_pos)
    #     input_text = input_text[:entity.start_pos] + mask + input_text[entity.end_pos:]
    #     # Adjust the end position based on the length difference
    #     entity.end_pos += length_diff
    
    for entity in entities:
        input_text = re.sub(fr"{entity.text}",f'*{entity.label}*',input_text.strip(),flags=re.I | re.MULTILINE)

    # Mask numbers (Including both integer and floating-point numbers)
    number_pattern = r'-?\d+(?:\.\d+)?'
    input_text = re.sub(number_pattern, "*NUMBER*", input_text)
    
    return input_text

# Function to perform NER
def perform_ner(request: InputText):
    input_text = request.text
    blacklist_data = [word.lower().strip() for word in request.blacklist.split(',')]
    doc = nlp(input_text)
    entities = []
    for ent in doc.ents:
        if ent.text.lower() not in blacklist_data:
            entities.append(
                    Entity(text=ent.text, start_pos=ent.start_char, end_pos=ent.end_char, label=ent.label_)
                )
    masked_output = mask_patterns(input_text, entities)
    return {"text": masked_output, "entities": entities}

# Endpoint to handle the POST request for NER
@app.post("/ner/")
async def ner_handler(request: InputText):
    return perform_ner(request)

# Mount the static directory to serve the HTML page
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates for rendering the HTML page
templates = Jinja2Templates(directory="static")

# Route to serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
