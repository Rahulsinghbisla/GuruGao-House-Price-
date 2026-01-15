from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from typing_extensions import TypedDict,Annotated,Literal
import operator
from langgraph.graph import StateGraph,START,END


load_dotenv()

class llm_state(TypedDict):
   next : Literal["GeneralAgent","PropertyAgent"]
   messages : Annotated[list[BaseMessage],operator.add]


checkpointer = InMemorySaver()

graph = StateGraph(llm_state)

def supervisior(state:llm_state):
   """You are a supervisor router for a chatbot.

      Your task is to analyze the user's message and decide which agent
      should handle it.

      Routing rules:
      - If the message is casual, conversational, greeting, or unrelated to property prices,
      route to GeneralAgent.
      - If the message is related to real estate, property prices, house cost prediction,
      buying/selling property, location-based pricing, or property features,
      route to PropertyAgent.

      Return ONLY the agent name.
"""

   user_input = state['messages']
   print("Supervisior")

   prompt = """You are a Supervisor Agent responsible for routing user queries to the correct agent.

               Your task:
               Analyze the user's query and decide which agent should handle it.

               Available agents:
               1. GeneralAgent  
                  - Casual conversation
                  - Greetings
                  - General questions
                  - Explanations not related to property data

               2. PropertyAgent  
                  - Property details
                  - Real estate information
                  - Property prices, locations, amenities
                  - Buying, selling, renting properties
                  - Queries mentioning flats, houses, apartments, plots, society names

               Routing rules:
               - If the query is related to real estate or property information, return: PropertyAgent
               - Otherwise, return: General

               Output rules:
               - Return ONLY one of the following values:
               - General
               - PropertyAgent
               - Do NOT provide explanations
               - Do NOT add extra text

               User query:
               {user_input}
"""

   agent =  create_agent(
    model = ChatOpenAI(model="gpt-5-nano"),
    system_prompt=prompt,
)
   next = agent.invoke({"messages": user_input})['messages'][-1].content
   return {'next':next}

def GeneralAgent(state:llm_state):
   """The GeneralAgent handles general conversation and non-property-related queries.

         It should answer:
         - Greetings (hi, hello, how are you)
         - Casual conversation
         - Questions about the chatbot
         - Small talk
         - General knowledge questions not related to property or real estate

         It should NOT:
         - Predict house prices
         - Answer real estate or property-related questions
"""
   
   user_input = state['messages']

   prompt="""You are a General AI Assistant.
      Your role:
      - Handle general conversation and non-property-related queries.
      - Provide clear, accurate, and concise responses.
      - Maintain a friendly and professional tone.

      You can help with:
      - Greetings and casual conversation
      - General knowledge questions
      - Explanations of concepts
      - Guidance, tips, and how-to questions
      - Non-technical and technical questions that are NOT related to property or real estate data

      Important rules:
      - If the user asks about property details, prices, locations, amenities, or real estate advice, DO NOT answer it.
      - Instead, respond with a brief message indicating that the query should be handled by the PropertyAgent.

      Response guidelines:
      - Be direct and easy to understand.
      - Avoid unnecessary verbosity.
      - Do not mention internal agents, routing, or system design.
      - Do not hallucinate facts.
      - If unsure, say you do not know.

      User query:
      {user_input}
      """
   agent = create_agent(
      model = ChatOpenAI(model="gpt-5-nano"),
      system_prompt=prompt,
   )
   res = agent.invoke({"messages": user_input})['messages'][-1].content
   return {'messages':[AIMessage(content=res)]}

def PropertyAgent(state:llm_state):
   """The PropertyAgent is responsible for handling all real estate and property-related queries.
      It should answer:
      - House price prediction
      - Property valuation
      - Buying or selling property questions
      - Location-based property pricing
      - Questions involving size, BHK, area, budget, society, amenities

      It can:
      - Ask for missing details (location, area, BHK, budget)
      - Use the trained ML model for predictions

      It should NOT:
      - Engage in casual conversation unrelated to property
"""
   user_input = state['messages']
   prompt ="""You are PropertyAgent, an expert real estate assistant specialized in
         house price prediction and property-related guidance.

         Your responsibilities:
         - Handle ONLY property and real estate related queries.
         - Assist users with house price prediction using a trained ML model.
         - Guide users in buying, selling, or evaluating residential properties.

         You should handle queries related to:
         - House price prediction
         - Flat / apartment / house valuation
         - Location-based property pricing
         - Budget-based property suggestions
         - Property features such as BHK, area (sqft), society, amenities, and location
         - Real estate trends and price estimation logic

         Conversation rules:
         - If the user asks for a price prediction, first check whether all required inputs are available.
         - Required inputs may include: location, area (sqft), BHK, property type, and budget (if applicable).
         - If any required information is missing, ask clear and specific follow-up questions.
         - Ask ONLY for missing information, not all details at once.
         - Keep responses short, professional, and helpful.

         Prediction rules:
         - Do NOT guess or fabricate prices.
         - Use the ML prediction tool or pipeline when available.
         - If prediction is not possible due to missing or invalid data, clearly explain why.

         Restrictions:
         - Do NOT engage in casual conversation or small talk.
         - Do NOT answer general knowledge or non-property-related questions.
         - If a query is outside real estate scope, politely ask the user to rephrase or redirect.

         Tone & Style:
         - Professional, polite, and confident
         - Explain predictions in simple terms
         - Avoid technical ML jargon unless asked

         Goal:
         Provide accurate, reliable, and user-friendly property price insights
         that help users make informed real estate decisions.
      User query:
      {user_input}
"""
   agent = create_agent(
      model = ChatOpenAI(model="gpt-5-nano"),
      system_prompt=prompt,
   )
   res = agent.invoke({"messages": user_input})['messages'][-1].content
   return {'messages':[AIMessage(content=res)]}

def condition_decide(state:llm_state):
   next = state['next']
   if next == "General":
      return "GeneralAgent"
   else :
      return "PropertyAgent"

graph.add_node("Supervisior",supervisior)
graph.add_node("GeneralAgent",GeneralAgent)
graph.add_node("PropertyAgent",PropertyAgent)

graph.add_edge(START,"Supervisior")
graph.add_conditional_edges("Supervisior",condition_decide)
graph.add_edge("PropertyAgent",END)
graph.add_edge("GeneralAgent",END)

agent = graph.compile(checkpointer=checkpointer)

# messages = [HumanMessage(content="Add 3 and 4.")]
# messages = workflow.invoke({"messages": messages})
# for m in messages["messages"]:
#     m.pretty_print()