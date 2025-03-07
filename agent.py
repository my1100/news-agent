import requests
import json
import getpass
import os
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition

# API configuration
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key:\n")
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

class State(TypedDict):
    messages: Annotated[List[Dict[str, str]], add_messages]  # Store conversation messages

# Create a state graph
graph_builder = StateGraph(State)

# Load LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,  # Adjust as needed
    max_tokens=5000,   # Adjust as needed
    timeout=None,     # Optional: Set a timeout for API calls
    max_retries=2,    # Optional: Set max retries for API calls
)

# Define search tool
tool = TavilySearchResults(max_results=2)  # Fetch up to 3 results
tools = [tool]

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    print("Reached chatbot")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Add chatbot node
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
print("Reached tool_node")

graph_builder.add_edge(START, "chatbot")  # Entry point
graph_builder.add_edge("chatbot", END)  # Exit point

# Add conditional edges
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Add an edge from tool node back to chatbot
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

# Create web interface with Flask
app = Flask(__name__)

# Global variable to store chat history
chat_history = []

@app.route("/chat", methods=["POST"])
def chat():
    """Handle user input and return AI response."""
    print("Reached chat")
    global chat_history

    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    chat_history.append(HumanMessage(content=user_input))

    # Invoke the graph to get AI response
    state = {"messages": chat_history}
    updated_state = graph.invoke(state, {"recursion_limit": 10})

    chat_history = updated_state["messages"]

    return jsonify({"response": chat_history[-1].content})

@app.route("/")
def home():
    """Render the chat interface."""
    return '''
    <html>
    <head>
        <title>Research Assistant</title>
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
        <h2>Newsagent</h2>
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