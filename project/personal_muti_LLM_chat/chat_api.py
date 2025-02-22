import fastapi
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
import os

get_database_uri=os.getenv("MONGODB_URI")
app = fastapi.FastAPI()
# Connect to MongoDB
client = MongoClient(get_database_uri)
db = client["chat_db"]
collection = db["chats"]


@app.post("/")
def create_chat(item: dict):
    user_message = item.get("user_message")
    ai_response = item.get("ai_response")
    
    # Save to MongoDB
    chat_data = {
        "user_message": user_message,
        "ai_response": ai_response
    }
    collection.insert_one(chat_data)
    
    return {"message": "Chat saved successfully"}

@app.get("/")
def get_chats():
    chats = list(collection.find())
    return {"chats": chats}