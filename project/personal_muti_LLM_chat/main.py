import streamlit as st
import os
import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import chat_api
import asyncio

async def main():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    # Model mapping dictionary
    MODEL_MAPPING = {
        "DeepSeek R1": "deepseek-r1-distill-llama-70b",
        "Llama 8B": "llama3-8b-8192",
        "Llama 70B": "llama3-70b-8192",
        "Gemini": "gemma2-9b-it",
        "Mixtral": "mixtral-8x7b-32768"
    }

    # Model descriptions
    MODEL_DESCRIPTIONS = {
        "DeepSeek R1": "DeepSeek R1 is a powerful LLaMA-based model optimized for efficiency and high-quality responses.",
        "Llama 8B": "Llama 8B is a mid-sized model providing balanced performance and accuracy for various AI applications.",
        "Llama 70B": "Llama 70B is an advanced model with enhanced capabilities for reasoning and detailed answers.",
        "Gemini": "Gemini (Gemma2-9B) is a fine-tuned model designed for high-quality conversational AI.",
        "Mixtral": "Mixtral 8x7B is a mixture of experts model providing exceptional performance on large-scale tasks."
    }

    # Streamlit Page Configuration
    st.set_page_config(page_title="All in One Chatbot", layout="centered")

    # Initialize session state with a default value
    if "session_value" not in st.session_state:
        st.session_state.session_value = "inam"

    if "session_auth" not in st.session_state:  
        st.session_state.session_auth = ""

    # Sidebar Configuration
    with st.sidebar:
        st.title("Chatbot Settings")

        selected_model = st.selectbox("Select Model", list(MODEL_MAPPING.keys()))
        model_name = MODEL_MAPPING[selected_model]
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max Tokens", 100, 4096, 1024)
        top_p = st.slider("Top-P", 0.0, 1.0, 1.0)
        frequency_penalty = st.slider("Frequency Penalty", 0.0, 1.0, 0.0)
        
        if st.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": f"Hi, I'm All in One ChatModel! I use different AI models. You've selected **{selected_model}**."}
            ]
            st.rerun()
        
        data = chat_api.get_chats()
        selected_chat = st.selectbox("Select a chat", [chat["_id"] for chat in data["chats"]])
        
        if selected_chat:
            chat_data = next(chat for chat in data["chats"] if chat["_id"] == selected_chat)
            st.write(f"Chat ID: {chat_data['_id']}")
            st.write(f"User: {chat_data['user_message']}")
            
            # Ensure messages key exists in chat_data
            if "messages" in chat_data:
                for message in chat_data["messages"]:
                    role = "user" if message["role"] == "user" else "assistant"
                    avatar = "👤" if role == "user" else "🤖"
                    with st.chat_message(role, avatar=avatar):
                        st.markdown(message["content"])

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", 
            "content": f"Hi, I'm All in One ChatAPP! I use different AI models. You've selected **{selected_model}**.",
            }
        ]

    # Check session state and display chat if session value is entered
    session_value = st.text_input("Enter session value:", st.session_state.session_auth)
    if session_value != "inam":
        st.warning("Please enter the correct session value to view the chat.")
        return

    st.session_state.session_value = session_value
    st.write(f"Session Value: {session_value}")

    data = chat_api.get_chats()
    data = list(data["chats"])
    
    for chat in data:
        if len(data) > 1:
            with st.chat_message("user", avatar="👤"):
                st.markdown(chat.get("user_message"))
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(chat.get("ai_response"))
        if len(data) > 20:
            # Only display the 10 most recent chats
            chat = chat[-10:]

    # Display chat history with formatted response
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)
            if message["role"] == "assistant" and "caption" in message:
                st.caption(message["caption"])

    # Define prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are {model_name}, a powerful AI model. 
        {model_description}
        
        Answer the question to the best of your ability, even if no additional context is provided.
        Provide the most accurate response based on the question. and and only response Minglish language 
        
        <context>
        {context}
        </context>
        
        Question: {input}
        """
    )

    # Chat input field
    if user_prompt := st.chat_input("How can I help you?"):
        
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_prompt)
        
        try:
            llm = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty
            )

            document_chain = create_stuff_documents_chain(llm, prompt_template)
            
            start = time.process_time()
            response = document_chain.invoke({
                "input": user_prompt,
                "model_name": selected_model,
                "model_description": MODEL_DESCRIPTIONS[selected_model],
                "context": "",
            })
            elapsed_time = time.process_time() - start
            
            bot_response = response
            res = chat_api.create_chat({"user_message": user_prompt, "ai_response": bot_response})
            st.write(res)
            print(bot_response)
            
            # Store assistant response in session state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response,
                "caption": f"🟡 *Response generated by {selected_model}* – ⏳ {elapsed_time:.2f} sec"
            })
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(bot_response, unsafe_allow_html=True)
                st.caption(f"🟡 *Response generated by {selected_model}* – ⏳ {elapsed_time:.2f} sec")

        except Exception as e:
            st.error(f"⚠️ Error generating response: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())