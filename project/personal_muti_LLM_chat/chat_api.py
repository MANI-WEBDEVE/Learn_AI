import fastapi
from pymongo import MongoClient

app = fastapi.FastAPI()
# Connect to MongoDB
client = MongoClient("mongodb+srv://inamkhan:inamchat@chat.xmdb1.mongodb.net/")
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