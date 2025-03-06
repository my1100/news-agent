import requests
import json
import os
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_deepseek import ChatDeepSeek
from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, SystemMessage

# DeepSeek API configuration
os.environ["DEEPSEEK_API_KEY"] = "sk-e2d15161b9a34264926b9e803b957061"  # Replace with your actual API key
ENDPOINT = "https://api.deepseek.com/v1"

# Define the Conversation State
class State(TypedDict):
    messages: Annotated[List[Dict[str, str]], add_messages]  # Store conversation messages

# Create a state graph
graph_builder = StateGraph(State)

# Load DeepSeek LLM
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,  # Adjust as needed
    max_tokens=500,   # Adjust as needed
    timeout=None,     # Optional: Set a timeout for API calls
    max_retries=2,    # Optional: Set max retries for API calls
)

def chatbot(state: State):
    """Process user messages with DeepSeek AI."""
    # Extract the latest user message
    return {"messages": [llm.invoke(state["messages"])]}

    # latest_message = state["messages"][-1]
    # if isinstance(latest_message, HumanMessage):
    #     user_input = latest_message.content  # Extract content from HumanMessage
    # else:
    #     user_input = latest_message["content"]  # Fallback for dictionary format
    #
    # # Get AI response
    # response = llm.invoke(state["messages"])
    #
    # # Add AI response to history
    # state["messages"].append(AIMessage(content=response.content))
    #
    # return state

# Add chatbot node
graph_builder.add_node("chatbot", chatbot)

# Connect the Graph
graph_builder.add_edge(START, "chatbot")  # Entry point
graph_builder.add_edge("chatbot", END)  # Exit point

# Compile the graph
graph = graph_builder.compile()

# Create Web Interface with Flask
app = Flask(__name__)

# Global variable to store chat history (for simplicity; use a database in production)
chat_history = []

@app.route("/chat", methods=["POST"])
def chat():
    """Handle user input and return AI response."""
    global chat_history

    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Add user message to chat history
    chat_history.append(HumanMessage(content=user_input))

    # Invoke the graph to get AI response
    state = {"messages": chat_history}
    updated_state = graph.invoke(state)

    # Update chat history with AI response
    chat_history = updated_state["messages"]

    # Return the latest AI response
    return jsonify({"response": chat_history[-1].content})

@app.route("/")
def home():
    """Render the chat interface."""
    return '''
    <html>
    <head>
        <title>Virtual Assistant</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            #chat-box {
                width: 400px;
                height: 500px;
                border: 1px solid #ccc;
                padding: 10px;
                overflow-y: auto;
                margin-bottom: 10px;
            }
            #user_input {
                width: 300px;
                padding: 5px;
            }
            button {
                padding: 5px 10px;
            }
        </style>
    </head>
    <body>
        <h2>DeepSeek Chat</h2>
        <div id="chat-box"></div>
        <input type="text" id="user_input" placeholder="Type a message...">
        <button onclick="sendMessage()">Send</button>
        <script>
            function sendMessage() {
                var user_input = document.getElementById("user_input").value;
                if (!user_input) return;

                // Add user message to chat box
                var chatBox = document.getElementById("chat-box");
                chatBox.innerHTML += `<p><strong>You:</strong> ${user_input}</p>`;

                // Clear input field
                document.getElementById("user_input").value = "";

                // Send message to server
                fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: user_input })
                })
                .then(response => response.json())
                .then(data => {
                    // Add AI response to chat box
                    chatBox.innerHTML += `<p><strong>AI:</strong> ${data.response}</p>`;
                    chatBox.scrollTop = chatBox.scrollHeight;  // Auto-scroll to bottom
                })
                .catch(error => {
                    console.error("Error:", error);
                });
            }
        </script>
    </body>
    </html>
    '''


if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)