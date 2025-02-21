
from fastapi import FastAPI, HTTPException
from mongoengine import connect, Document, StringField, FloatField, DoesNotExist
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# MongoDB connection with mongoengine
try:
    connect(db="mydatabase", host=os.getenv("MONGODB_URI"))
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")


# Mongoengine document model
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

    meta = {'collection': 'items'}

# Pydantic model for data validation
class Item(BaseModel):
    ip: str
    city: Optional[str] = None
    region: str
    country: str
    loc: str
    org: Optional[str] = None
    postal: str
    timezone: Optional[str] = None
    readme: Optional[str] = None

# Convert Document to dict
def item_helper(item):
    return {
        "id": str(item.id),
        "ip": item.ip,
        "city": item.city,
        "region": item.region,
        "country": item.country,
        "loc": item.loc,
        "org": item.org,
        "postal": item.postal,
        "timezone": item.timezone,
        "readme": item.readme
    }

@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI with MongoDB"}

@app.post("/")
async def create_loc(item: Item):
    try:
        item_doc = ItemDocument(**item.dict())
        item_doc.save()
        return item_helper(item_doc), "Success", 200
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
