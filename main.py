from dotenv import load_dotenv 
from typing import Annotated, Literal 
from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages
from pydantic import BaseModel,Field
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
load_dotenv()

llm = ChatGroq(
    model_name="llama3-70b-8192",  
    temperature=0.7
)
class MessageClassifier(BaseModel):
    message_type: Literal["mental","physical"] = Field(
        ...,
        description="Classify if the message is about mental or physical problems."
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

def classify_message(state:State):
    last_msg = state['messages'][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)
    result = classifier_llm.invoke([
        {
            "role": 'system',
            "content" : """ You are a message classifier for a wellness chatbot.
                    Classify the user's message as one of the following types:
                    - "mental": If the message is about emotions, stress, anxiety, depression, motivation, feeling overwhelmed, or mental well-being in general.
                    - "physical": If the message is about exercise, fitness, nutrition, sleep, energy levels, or general physical health.
                    Only respond with one of the two labels: 'mental' or 'physical'
            """
        },
        {"role":"user","content": last_msg.content}
    ])
    return {"message_type": result.message_type}

def mental_agent(state:State):
    last_msg = state['messages'][-1] 
    messages = [
        {"role":"system",
         "content":  """You are a gentle and supportive mental wellness coach.
                        Offer short emotional support, encouragement, or mindfulness suggestions.
                        Keep your response under 3 sentences. Be warm and concise.
                        Do not give clinical diagnoses or medical advice."""
         },
        {
            "role": "user",
            "content": last_msg.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages":[{"role":"assistant","content":reply.content}]}

def physical_agent(state:State):
    last_msg = state['messages'][-1] 
    messages = [
        {"role":"system",
         "content": """ You are a helpful physical wellness coach.
                        Give short, clear advice about fitness, nutrition, sleep, or healthy habits.
                        Keep your responses under 3 sentences. Be concise and avoid long explanations.
                        Do not diagnose medical issues — focus on general healthy lifestyle suggestions.
                        """
         },
        {
            "role": "user",
            "content": last_msg.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages":[{"role":"assistant","content":reply.content}]}
 
def router(state:State):
    message_type = state.get('message_type','physical')
    if message_type == 'mental':
        return {"next":"mental"}
    return {"next":"physical"}

graph = StateGraph(State)
graph.add_node("classifier",classify_message)
graph.add_node("router",router)
graph.add_node("mental",mental_agent)
graph.add_node("physical",physical_agent)

graph.add_edge(START,"classifier")
graph.add_edge("classifier","router")

graph.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"mental": "mental", "physical": "physical"}
)


graph.add_edge("mental",END)
graph.add_edge("physical",END)

graph = graph.compile()

def run_chatbot():
    state = {"messages": [], "message_type": None}
    while (user_input := input("\nYou: ")) != "exit":
        state["messages"].append({"role": "user", "content": user_input})
        state = graph.invoke(state)
        if state.get("messages"):
            assistant_response = state["messages"][-1].content
            print("\n\033[1;32mAssistant:\033[0m", assistant_response)
    print("\n\033[1;31mByee.\033[0m")


if __name__ == "__main__":
    run_chatbot()
