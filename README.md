# Wellness ChatBot with LangGraph

This project is a wellness-focused chatbot built using [LangGraph](https://docs.langgraph.dev/) with [Groq LLMs](https://console.groq.com/) LLM. It classifies user messages into **mental** or **physical** wellness topics and responds accordingly using two different assistant agents.

## Features

- Classifies user input as **mental** or **physical** wellness topics
- Responds with either:
  - Emotional and supportive responses for mental wellness
  - Practical advice for physical wellness
- Uses `llama3-70b-8192` via Groq for fast, low-latency inference
- <img width="823" alt="Screenshot 2025-06-29 at 9 47 37 PM" src="https://github.com/user-attachments/assets/0a726c8e-452c-433e-a919-e971d3704d62" />


## Technologies Used

- **LangGraph**: For building and routing the conversational workflow
- **LangChain**: For interacting with language models and managing message history
- **Groq (LLM Provider)**: For high-speed inference using LLaMA 3 model

## Project Structure

- `classify_message`: Classifies the user message as "mental" or "physical"
- `mental_agent`: Responds to mental health-related queries
- `physical_agent`: Responds to physical health-related queries
- `router`: Directs the flow based on classification
- `StateGraph`: Handles the entire graph structure and transitions
- ![output](https://github.com/user-attachments/assets/29c2893b-82a4-4ffd-b2b8-3b676e70bd81)


## How It Works

1. User enters a message.
2. The message is classified as either "mental" or "physical".
3. The router directs it to the appropriate agent.
4. The selected agent generates a short, helpful reply.

