from fastapi import FastAPI, HTTPException
from mongoengine import connect, Document, StringField
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Connect to MongoDB
try:
    connect(db="mydatabase", host=os.getenv("MONGODB_URI"))
except Exception as e:
    print(f"MongoDB Connection Error: {e}")

# MongoDB Schema
class ItemDocument(Document):
    ip = StringField(required=True)
    city = StringField()
    region = StringField(required=True)
    country = StringField(required=True)
    loc = StringField(required=True)
    org = StringField()
    postal = StringField(required=True)
    timezone = StringField()
    readme = StringField()
    
    meta = {'collection': 'locations'}

# Pydantic Model for Validation
class Item(BaseModel):
    ip: str
    city: str | None = None
    region: str
    country: str
    loc: str
    org: str | None = None
    postal: str
    timezone: str | None = None
    readme: str | None = None

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI MongoDB Service"}

@app.post("/store-location")
def create_loc(item: Item):
    try:
        item_doc = ItemDocument(**item.dict())
        item_doc.save()
        return {"status": "success", "data": item.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
