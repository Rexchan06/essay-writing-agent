import os
from dotenv import load_dotenv
load_dotenv() 

from langchain.chat_models import init_chat_model

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  # Add this to your .env file
os.environ["LANGCHAIN_PROJECT"] = "ESSAY-WRITING-AGENT"  # Optional: organize your traces

llm = init_chat_model("google_genai:gemini-2.0-flash")

from typing import Any, List
from typing_extensions import Dict

from langchain_tavily import TavilySearch

from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

#Defining the State
class State(TypedDict):
    #Input & Configuration
    messages: Dict
    topic: str
    requirements: Dict[str, Any]
    user_preferences: Dict[str, Any]

    #Research Phase
    research_query_history: List[str]
    research_query: str
    research_results: Dict[str, Any]
    current_query: str
    source_inventory: List[Dict]
    research_gaps: List[str]

    #Planning Phase
    thesis_statement: str
    argument_map: Dict[str, Any]
    essay_outline: List[Dict]
    section_dependencies: Dict[str, List[str]]

    #Writing Phase
    draft_sections: Dict[str, Dict]
    cross_reference: Dict[str, List[str]]
    writing_coordination: Dict[str, Any]

    #Review Phase
    review_results: Dict[str, Dict]
    revision_queue: List[Dict]
    quality_metric: Dict[str, float]

    #Integration Phase
    assembly_essay: str
    reference_list: List[Dict]
    final_formatting: Dict[str, Any]

    #System Coordination
    workflow_status: str
    agent_communication: List[Dict]
    error_log: List[Dict]
    checkpoint_data: Dict[str, Any]

    should_continue: bool
    summarised_result: str

def determine_user_requirement(state: State):
    """Determine user topic, requirements & user preference"""
    topic_prompt = PromptTemplate.from_template("""
    Based on the user query: {query}, determine the topic of the essay that user wants to write about. 
    Only return the topic name. Nothing else.
    """)
    topic_message = topic_prompt.format(query=state["messages"][-1]["content"])
    topic = llm.invoke([HumanMessage(content=topic_message)])
    
    requirement_prompt = PromptTemplate.from_template("""
    Based on the user query: {query}, determine the CONTENT requirements of the essay.
    Content requirements refer to what information or topics must be covered in the essay.
    For example, requirements might include "historical events", "analysis of causes", "comparison of theories", etc.
    
    Style preferences like "writing in Shakespeare's style" are NOT content requirements.
    
    Only return the content requirements. If no specific content requirements, return "No specific content requirements".
    """)
    requirement_message = requirement_prompt.format(query=state["messages"][-1]["content"])
    requirement = llm.invoke([HumanMessage(content=requirement_message)])
    
    preference_prompt = PromptTemplate.from_template("""
    Based on the user query: {query}, determine the user's STYLE PREFERENCES for this essay.
    Style preferences refer to HOW the essay should be written, not what content it should contain.
    
    Examples of style preferences include:
    - Writing in a specific author's style (e.g., "Shakespeare's writing style")
    - Tone preferences (formal, casual, humorous)
    - Structure preferences (narrative, analytical)
    
    Only return the style preferences. If no style preferences mentioned, return "No specific style preferences".
    """)
    preference_message = preference_prompt.format(query=state["messages"][-1]["content"])
    preference = llm.invoke([HumanMessage(content=preference_message)])
    
    return {"topic": topic.content, "requirements": requirement, "user_preferences": preference}

def research_agent(state: State):
    """Do Research On Specific Topic"""
    # Initialize keys if they don't exist
    if "research_query_history" not in state:
        state["research_query_history"] = []
    if "research_results" not in state:
        state["research_results"] = []
        
    prompt_template = PromptTemplate.from_template("""
    Based on the user's topic: {topic} and user's requirements: {requirements} for the essay, 
    write a query to search using TavilySearch. 
    
    These are the research query history: {research_query_history}
    These are the research results so far: {research_results}
    
    Write a specific search query that will help gather additional information we still need. Do not write out your thoughts, only the query.
    If you believe we have sufficient information to address all the requirements, respond with "RESEARCH_COMPLETE".
    """)

    prompt_message = prompt_template.format(topic=state["topic"], requirements=state["requirements"], research_query_history=state["research_query_history"], research_results=state["research_results"])
    response = llm.invoke(prompt_message)

    if "RESEARCH_COMPLETE" in response.content:
        return {"should_continue": False}

    state["research_query_history"].append(response.content)
    state["current_query"] = response
    state["should_continue"] = True

    return state

def tavily_search(state: State):
    """Do research on specific topic"""
    query = state["current_query"]
    query_text = query.content if hasattr (query, "content") else str(query)
    query_text = query_text.replace('"', '').replace("'", "").strip()
    tavily_search = TavilySearch(max_result=2, topic="general")
    results = tavily_search.invoke(query_text)
    
    # Add results to the state
    if "research_results" not in state:
        state["research_results"] = []
    state["research_results"].append({"query": query, "results": results})

    return state

def summarise_findings(state: State):
    research_results = state["research_results"]
    prompt_template = PromptTemplate.from_template("""Based on the research_results: {research_results}, write me a summary of what you learnt""")
    prompt_message = prompt_template.format(research_results=research_results)
    response = llm.invoke(prompt_message)
    
    state["summarised_result"] = response.content
    return state

from json import loads

def planning_agent(state: State):
    """Plan the essay structure and outline"""
    research_results = state["research_results"]
    prompt_template = PromptTemplate.from_template("""Based on the research results: {research_results}, the user topic: {topic}, the user requirements: {requirements}, write a thesis statement for this essay.""")
    prompt_message = prompt_template.format(research_results=research_results, topic=state["topic"], requirements=state["requirements"])
    thesis_statement = llm.invoke(prompt_message)
    state["thesis_statement"] = thesis_statement.content

    outline_prompt_template = PromptTemplate.from_template("""Create a detailed essay outline based on thesis statement: {thesis_statement}, research results: {research_results}, the user topic: {topic}, the user requirements: {requirements}. 

        Return as JSON array where each object represents a paragraph with this exact structure:

        [
        {{
            "paragraph_id": "introduction",
            "paragraph_type": "introduction",
            "main_point": "Clear thesis statement that directly addresses the topic",
            "key_elements": [
            "Hook to grab reader attention",
            "Background context on the topic", 
            "Thesis statement with main argument"
            ],
            "supporting_details": [
            "Specific hook example or statistic",
            "Key background information to include",
            "Preview of main points to be discussed"
            ],
            "research_integration": [
            "Specific research findings to mention in introduction",
            "Statistics or data points to include"
            ]
        }},
        {{
            "paragraph_id": "body_1",
            "paragraph_type": "body",
            "main_point": "First major argument supporting the thesis",
            "topic_sentence": "Clear topic sentence for this paragraph",
            "key_elements": [
            "Primary argument or claim",
            "Evidence supporting the claim",
            "Analysis connecting evidence to thesis"
            ],
            "supporting_details": [
            "Specific examples or case studies",
            "Data points or statistics",
            "Expert quotes or testimonials"
            ],
            "research_integration": [
            "Specific research findings relevant to this point",
            "How research supports the main argument",
            "Key sources to cite"
            ],
            "transitions": {{
            "opening": "How to connect from previous paragraph",
            "closing": "How to lead into next paragraph"
            }}
        }},
        {{
            "paragraph_id": "body_2", 
            "paragraph_type": "body",
            "main_point": "Second major argument supporting the thesis",
            "topic_sentence": "Clear topic sentence for this paragraph",
            "key_elements": [
            "Primary argument or claim",
            "Evidence supporting the claim", 
            "Analysis connecting evidence to thesis"
            ],
            "supporting_details": [
            "Specific examples or case studies",
            "Data points or statistics",
            "Expert quotes or testimonials"
            ],
            "research_integration": [
            "Specific research findings relevant to this point",
            "How research supports the main argument",
            "Key sources to cite"
            ],
            "transitions": {{
            "opening": "How to connect from previous paragraph",
            "closing": "How to lead into next paragraph"
            }}
        }},
        {{
            "paragraph_id": "conclusion",
            "paragraph_type": "conclusion",
            "main_point": "Restatement of thesis with final thoughts",
            "key_elements": [
            "Restate thesis in new words",
            "Summarize main supporting points",
            "Provide final insight or call to action"
            ],
            "supporting_details": [
            "Brief recap of strongest evidence",
            "Implications of the arguments presented",
            "Future considerations or recommendations"
            ],
            "research_integration": [
            "Overall research conclusions",
            "Broader implications from research findings"
            ]
        }}
        ]

        Important guidelines:
        1. Create exactly the number of body paragraphs needed based on the requirements
        2. Each paragraph should have a clear, specific focus
        3. Integrate research findings naturally into each paragraph plan
        4. Ensure logical flow from introduction through body paragraphs to conclusion
        5. Make sure all paragraphs support the central thesis statement
        6. Include specific, actionable details that will help with paragraph writing

        Topic: {topic}
        Requirements: {requirements}
        Thesis: {thesis_statement}
        Research: {research_results}

        Return only the JSON array, no additional text.""")

    outline_prompt_message = outline_prompt_template.format(
        topic=state["topic"],
        requirements=state["requirements"],
        thesis_statement=thesis_statement,
        research_results=research_results
    )

    state["essay_outline"] = loads(llm.invoke(outline_prompt_message))

    return state

def parallel_writing_agent(essay_outline: List[Dict]) -> Dict[str, str]:
    """Write all paragraphs in parallel"""

    parallel_tasks = {}
    inputs = {}

    for para_info in essay_outline:
        paragraph_id = para_info["paragraph_id"]
        paragraph_type = para_info["paragraph_type"]

        if paragraph_type == "introduction":
            parallel_tasks[paragraph_id] = 

def tools_condition(state: State):
    """Determines if we should continue using tools"""
    return state.get("should_continue", True)

graphbuilder = StateGraph(State)
graphbuilder.add_node("determine_user_requirement", determine_user_requirement)
graphbuilder.set_entry_point("determine_user_requirement")
graphbuilder.add_node("research_agent", research_agent)
graphbuilder.add_edge("determine_user_requirement", "research_agent")
graphbuilder.add_node("tavily_search", tavily_search)
# graphbuilder.add_node("summarise", summarise_findings)
graphbuilder.add_node("planning_agent", planning_agent)
graphbuilder.add_conditional_edges(
   "research_agent",
   tools_condition,
   {
       True: "tavily_search",
       False: "planning_agent" 
   } 
)
graphbuilder.add_edge("tavily_search", "research_agent")

graph = graphbuilder.compile()

user_input = "Write me an essay about the fall of the Roman Empire. I need you to cover the history, cause of the fall and the consequences of it. I prefer Shakespeare's writing style."
events = graph.stream({"messages": [{"role": "user", "content": user_input}]}, stream_mode="values")

for event in events:
    if "messages" in event and event["messages"]:
        latest_message = event["messages"][-1]
        if latest_message["role"] == "system":
            print(f"[SYSTEM]: {latest_message['content']}")
        elif latest_message["role"] == "assistant":
            print(f"[ASSISTANT]: {latest_message['content']}")
        else:
            print(f"[{latest_message['role'].upper()}]: {latest_message['content']}")
    elif "research_results" in event and event["research_results"]:
        print(f"[RESEARCH]: Found {len(event['research_results'])} result sets")
    elif "assembly_essay" in event and event["assembly_essay"]:
        print(f"[FINAL ESSAY]:\n{event['assembly_essay']}")
    else:
        print("[EVENT]:", event)