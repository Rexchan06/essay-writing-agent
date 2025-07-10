import os
from dotenv import load_dotenv
import time
load_dotenv() 

from langchain.chat_models import init_chat_model
from langchain_together import ChatTogether

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")
# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  # Add this to your .env file
os.environ["LANGCHAIN_PROJECT"] = "ESSAY-WRITING-AGENT"  # Optional: organize your traces

llm = init_chat_model("google_genai:gemini-2.0-flash")

mistral = ChatTogether(
    together_api_key=together_api_key,
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
)

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
    research_results: Dict[str, Any]
    current_query: str

    #Planning Phase
    thesis_statement: str
    essay_outline: List[Dict]

    #Writing Phase
    draft_sections: List[str]
    final_sections: List[str]

    #Review Phase
    reviewed_sections: List[str]
    naturalizer_feedback: str

    #Integration Phase
    assembly_essay: str
    reference_list: List[Dict]
    final_formatting: Dict[str, Any]

    should_continue: bool

def determine_user_requirement(state: State):
    """Determine user topic, requirements & user preference"""
    topic_prompt = PromptTemplate.from_template("""
    Based on the user query: {query}, determine the topic of the essay that user wants to write about. 
    Only return the topic name. Nothing else.
    """)
    topic_message = topic_prompt.format(query=state["messages"][-1]["content"])
    topic = llm.invoke([HumanMessage(content=topic_message)])
    
    requirement_prompt = PromptTemplate.from_template("""
        Based on the user query: {query}, determine the CONTENT and STRUCTURAL requirements of the essay.
        
        CONTENT REQUIREMENTS:
        - What specific topics, information, or arguments must be covered in the essay
        - Examples include "historical events", "analysis of causes", "comparison of theories", etc.
        
        STRUCTURAL REQUIREMENTS:
        - Word count: Extract any word count requirements (e.g., "750-word essay", "at least 1000 words")
        - Paragraph organization: Determine if main arguments and counterarguments should be in separate paragraphs or combined
        * If specified to combine them (e.g., "integrate counterarguments into each body paragraph"), note this
        * If specified to separate them (e.g., "include a separate section for counterarguments"), note this
        * If not specified, indicate "No specific instruction on argument organization"
        - Number of arguments: Note if a specific number of arguments is requested (e.g., "three main points")
        
        FORMAT YOUR RESPONSE AS:
        Content Requirements: [List the content requirements]
        Word Count: [Specify word count requirement or "Not specified"]
        Argument Organization: [Specify if arguments and counterarguments should be combined or separate]
        Number of Arguments: [Specify number or "Not specified"]
        
        Do NOT include style preferences like "writing in Shakespeare's style" or "formal tone" as these are not content requirements.
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
    
    return {"topic": topic.content, "requirements": requirement.content, "user_preferences": preference.content}

def research_agent(state: State):
    """Do Research On Specific Topic"""
    # Initialize keys if they don't exist
    print("Entering research_agent function")
    if "research_query_history" not in state:
        state["research_query_history"] = []
    if "research_results" not in state:
        state["research_results"] = []

    query_count = len(state["research_query_history"])
    print(f"Current query count: {query_count}")

    if len(state["research_query_history"]) >= 8:
        print("Query limit reached, stopping research")
        return {"should_continue": False}
        
    prompt_template = PromptTemplate.from_template("""
        RESEARCH QUERY GENERATION TASK
        TOPIC: {topic}
        ESSAY REQUIREMENTS: {requirements}
        PREVIOUS QUERIES:
        {research_query_history}
        INFORMATION GATHERED SO FAR:
        {research_results}
        INSTRUCTIONS:
        1. Analyze the essay requirements and information gaps in our current research
        2. Create ONE specific, focused search query to gather missing information
        3. Prioritize queries that will yield factual evidence, examples, or expert opinions
        4. Avoid general or redundant queries that overlap with previous searches
        5. Format your query for optimal search results (use quotes for exact phrases, avoid unnecessary words)
        CONSTRAINTS:
        - You have a maximum of 8 total queries for this research phase
        - You have used {query_count} queries so far
        - If you've reached 8 queries or have sufficient information, respond with "RESEARCH_COMPLETE"
        RESPONSE FORMAT:
        Return ONLY your search query or "RESEARCH_COMPLETE" - no explanations, thoughts, or additional text.
    """)

    query_count = len(state["research_query_history"])
    prompt_message = prompt_template.format(topic=state["topic"], requirements=state["requirements"], research_query_history=state["research_query_history"], research_results=state["research_results"], query_count=len(state["research_query_history"]))
    response = mistral.invoke(prompt_message)

    if "RESEARCH_COMPLETE" in response.content:
        print(f"Model returned RESEARCH_COMPLETE: {response.content}")
        return {"should_continue": False}

    if query_count >= 7:
        print("This is the 7th query, will stop after this one")
        state["research_query_history"].append(response.content)
        state["current_query"] = response
        return {"should_continue": False}
    
    print(f"Generated query: {response.content}")

    state["research_query_history"].append(response.content)
    state["current_query"] = response
    state["should_continue"] = True
    print(f"Setting should_continue to {state['should_continue']}")

    return state

def tavily_search(state: State):
    """Do research on specific topic"""
    print("Entering tavily_search function")
    
    query = state["current_query"]
    query_text = query.content if hasattr(query, "content") else str(query)
    query_text = query_text.replace('"', '').replace("'", "").strip()
    print(f"Searching for: {query_text}")
    
    try:
        tavily_search = TavilySearch(max_results=3, topic="general")
        print("TavilySearch instance created")
        results = tavily_search.invoke(query_text)
        print(f"Search completed, got results: {type(results)}")
        
        # Add results to the state
        if "research_results" not in state:
            state["research_results"] = []
            
        state["research_results"].append({
            "query": query_text,
            "results": results
        })
        print("Results added to state")
        
    except Exception as e:
        print(f"Tavily search error: {str(e)}")
        # Add mock results
        # mock_results = [{"title": "Mock result", "content": f"Mock content about {query_text}"}]
        
        # if "research_results" not in state:
        #     state["research_results"] = []
            
        # state["research_results"].append({
        #     "query": query_text,
        #     "results": mock_results
        # })
        # print("Mock results added to state")
    
    return state

# def summarise_findings(state: State):
#     research_results = state["research_results"]
#     prompt_template = PromptTemplate.from_template("""Based on the research_results: {research_results}, write me a summary of what you learnt""")
#     prompt_message = prompt_template.format(research_results=research_results)
#     response = llm.invoke(prompt_message)
    
#     state["summarised_result"] = response.content
#     return state

from json import loads

def planning_agent(state: State):
    """Plan the essay structure and outline"""
    if "research_results" not in state:
        state["research_results"] = []
        
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

        Return only the JSON array, no additional text.""")

    outline_prompt_message = outline_prompt_template.format(
        topic=state["topic"],
        requirements=state["requirements"],
        thesis_statement=thesis_statement,
        research_results=research_results
    )

    response = llm.invoke(outline_prompt_message)

    content = response.content
    if "```json" in content:
        # Extract the JSON part from the markdown code block
        json_content = content.split("```json")[1].split("```")[0].strip()
    else:
        # If not in code block, just use the content as is
        json_content = content
        
    state["essay_outline"] = loads(json_content)

    return state

def writing_agent(state: State):
    """Write all paragraphs"""
    if "draft_sections" not in state:
        state["draft_sections"] = []

    for paragraph_outline in state["essay_outline"]:
        time.sleep(5)
        prompt_template = PromptTemplate.from_template("""
            Write a detailed paragraph based on the following outline:
            {paragraph_outline}
            This paragraph should:
            1. Begin with the topic sentence specified in the outline
            2. Include all key elements and supporting details mentioned in the outline
            3. Incorporate relevant research points from the outline
            4. Maintain coherent transitions from/to adjacent paragraphs: {previous_paragraphs} (Do not repeat content from the previous paragraph)
            5. Support the overall thesis statement: {thesis_statement}
            6. Be written in {preference} writing style
            7. Do not write any inline citation of any sorts.                                        
            Your paragraph should be well-structured and engaging. Include real life examples, evidence, or analysis as indicated in the outline. Ensure the writing flows naturally while maintaining the specified style preferences.
            If this is an introduction or conclusion paragraph, make sure it serves its proper function (introducing the topic and thesis or summarizing key points and providing closure).
        """)
        prompt_message = prompt_template.format(paragraph_outline=paragraph_outline, previous_paragraphs=state["draft_sections"], thesis_statement=state["thesis_statement"], preference=state["user_preferences"])
        response = llm.invoke(prompt_message)
        state["draft_sections"].append(response.content)

    return state

def review_agent(state: State):
    """Review each paragraph"""
    if "reviewed_sections" not in state: 
        state["reviewed_sections"] = []

    for draft_paragraph in state["draft_sections"]:
        time.sleep(5)
        prompt_template = PromptTemplate.from_template("""
            REWRITE TASK: Review and rewrite the following paragraph to ensure it meets all quality criteria.
            ORIGINAL PARAGRAPH:
            {draft_paragraph}
            REQUIREMENTS:
            1. Ensure strong alignment with the thesis statement: "{thesis_statement}"
            2. Maintain the requested writing style: "{preference}"
            3. Include all necessary structural elements:
            - Clear topic sentence
            - Supporting evidence and examples
            - Logical flow and transitions. Entire draft essay for your reference: {draft_sections}
            - Proper integration of research
            4. Ensure high language quality:
            - Correct grammar and syntax
            - Appropriate vocabulary
            - Varied sentence structure
            - Clear and concise expression
            - Add rhetorical richness and depth in framing
            IMPORTANT INSTRUCTIONS:
            - If the paragraph already meets all criteria, reproduce it with minimal changes
            - Don't add any inline citation of any sorts
            - If improvements are needed, rewrite while preserving the original meaning and intent
            - Do not repeat content from previous paragraphs
            - Ensure the paragraph flows naturally from the preceding content
            - Maintain the appropriate paragraph length (neither too short nor excessively long)
            PROVIDE ONLY THE REWRITTEN PARAGRAPH AS YOUR RESPONSE, WITHOUT ANY ADDITIONAL COMMENTARY.
            """)
        
        prompt_message = prompt_template.format(draft_paragraph=draft_paragraph, thesis_statement=state["thesis_statement"], research_results=state["research_results"], preference=state["user_preferences"], draft_sections=state["draft_sections"])
        response = llm.invoke(prompt_message)

        state["reviewed_sections"].append(response.content)

    return state

def citation_insertor(state: State):
    if "final_sections" not in state:
        state["final_sections"] = []

    prompt_template = PromptTemplate.from_template("""
        CITATION TASK: Add appropriate academic citations to the following the following essay: {essay}
        
        AVAILABLE RESEARCH SOURCES:
        {research_results}
        
        INSTRUCTIONS:
        1. Identify which research result was used in the essay
        2. Cite the reference in APA format as a list
        3. Add the word "Reference List" before all the references
        
        IMPORTANT:
        - Use standard APA in-text citation format: (Author, Year) or (Organization, Year)
        - DO NOT use LaTeX-style citations
        - DO NOT add any bibliography, reference list, or explanatory notes at the end
        - DO NOT include the run IDs or other technical identifiers in citations
        - No additional commentary and don't return any part of the essay
    """)
    prompt_message = prompt_template.format(essay=state["reviewed_sections"], research_results=state["research_results"])
    response = mistral.invoke(prompt_message)
    state["reviewed_sections"].append(response.content)

    return state

def compile_essay(state: State):
    """Compile essay into one"""
    prompt_template = PromptTemplate.from_template("""
        ESSAY COMPILATION TASK
        SECTIONS TO COMPILE:
        {reviewed_sections}
        INSTRUCTIONS:
        1. Assemble all sections into a cohesive, flowing essay
        2. Ensure smooth transitions between paragraphs and sections
        3. Maintain consistent formatting throughout the document
        4. Preserve the writing style established in the individual sections
        5. Verify that the overall structure follows a logical progression:
        - Introduction with clear thesis
        - Body paragraphs that develop arguments sequentially
        - Conclusion that synthesizes key points
        6. Remove any redundancies or repetitions between sections
        7. Standardize citation format if references are included
        8. Ensure the essay reads as a unified whole rather than disconnected parts
        FORMAT REQUIREMENTS:
        - Include a title centered at the top
        - Maintain consistent paragraph spacing
        - Do not include section numbers or headers unless they were in the original sections
        - Preserve any formatting like italics or quotations from the original sections
        PROVIDE ONLY THE FINAL, POLISHED ESSAY WITHOUT ANY META-COMMENTARY, EXPLANATIONS, OR NOTES ABOUT THE COMPILATION PROCESS.
        """)
    prompt_message = prompt_template.format(reviewed_sections=state["reviewed_sections"])
    response = llm.invoke(prompt_message)
    state["assembly_essay"] = response.content

    return state

def naturalizer_feedback_generator(state: State):
    prompt_template = PromptTemplate.from_template("""
        Analyze the following essay and provide specific, actionable feedback:
        {assembly_essay}
        Please focus on:
        1. TONE: Identify any sections that sound overly formal, academic, or robotic. Suggest specific word choices or phrasing alternatives that would make these passages sound more natural and conversational.
        2. TRANSITIONS: Highlight areas where the flow between sentences or paragraphs feels abrupt or disconnected. Provide example transition phrases that would create smoother connections between ideas.
        3. CONCISENESS: Point out any redundant expressions, unnecessary qualifiers, or overly complex sentences. Suggest more concise alternatives without losing the original meaning.
        4. AUTHENTICITY: Identify specific sentences or paragraphs that lack a human voice. Recommend ways to incorporate more personality, varied sentence structures, or rhetorical techniques that would make the writing more engaging.
        For each suggestion, please:
        - Quote the specific text that needs improvement
        - Explain why it sounds mechanical or unnatural
        - Provide a rewritten example that maintains the same meaning but sounds more human
        Your feedback should be detailed enough to guide meaningful revisions while maintaining the essay's original arguments and structure.
        """)
    prompt_message = prompt_template.format(assembly_essay=state["assembly_essay"])
    response = mistral.invoke(prompt_message)
    state["naturalizer_feedback"] = response.content

    return state

def naturalizer_editor(state: State):
    prompt_template = PromptTemplate.from_template("""
        REVISION TASK: Transform the following essay based on the provided feedback.
        ORIGINAL ESSAY:
        {assembly_essay}
        FEEDBACK TO IMPLEMENT:
        {naturalizer_feedback}
        INSTRUCTIONS:
        1. Apply all suggested improvements for tone, transitions, conciseness, and authenticity
        2. Maintain the original argument structure and key points
        3. Ensure consistent voice throughout the revised version
        4. Keep approximately the same length as the original
        IMPORTANT: Return ONLY the revised essay with all improvements incorporated. Do not include explanations, comments, or any text outside the essay itself but remain the title on top.
        """)
    prompt_message = prompt_template.format(assembly_essay=state["assembly_essay"], naturalizer_feedback=state["naturalizer_feedback"])
    response = llm.invoke(prompt_message)
    state["assembly_essay"] = response.content

    return state

def redundancy_checker(state: State):
    prompt_template = PromptTemplate.from_template("""
    Revise the following essay to comply with a word count limit of {max_words} words if it is state in the requirement.
    Your goals:
    1. Remove any redundancy, repetition, and verbosity.
    2. Preserve all key arguments, transitions, and examples.
    3. Ensure the essay still flows logically and persuasively.
    4. Keep the tone natural and academic.

    Return the revised whole essay but make sure the title remains.

    Essay:
    {essay}
    """)
    prompt_message = prompt_template.format(max_words=state["requirements"], essay=state["assembly_essay"])
    response = llm.invoke(prompt_message)
    state["assembly_essay"] = response.content

    return state

def tone_polisher(state: State):
    prompt_template = PromptTemplate.from_template("""
        You are an expert academic editor. Improve the essay below while preserving its original argument and meaning. Focus on:
        - Enhancing formal academic tone (or {preference})
        - Making the language clearer, more concise, and natural
        - Write in a way that leaves an impact in other's mind
        - Be persuasive and confident
        - Improving coherence and flow across paragraphs
        - Fixing any awkward or repetitive phrasing
        Do not remove important arguments or examples. Aim to keep the length roughly the same.
        Essay:
        {essay}
        Return the revised whole essay and remain the title.
    """)
    prompt_message = prompt_template.format(preference=state["user_preferences"], essay=state["assembly_essay"])
    response = llm.invoke(prompt_message)
    state["assembly_essay"] = response.content

    return state

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
graphbuilder.add_node("writing_agent", writing_agent)
graphbuilder.add_node("review_agent", review_agent)
graphbuilder.add_node("citation_insertor", citation_insertor)
graphbuilder.add_node("compile_essay", compile_essay)
graphbuilder.add_node("naturalizer_feedback_generator", naturalizer_feedback_generator)
graphbuilder.add_node("naturalizer_editor", naturalizer_editor)
graphbuilder.add_node("redundancy_checker", redundancy_checker)
graphbuilder.add_node("tone_polisher", tone_polisher)
graphbuilder.add_conditional_edges(
   "research_agent",
   tools_condition,
   {
       True: "tavily_search",
       False: "planning_agent" 
   } 
)
graphbuilder.add_edge("tavily_search", "research_agent")
graphbuilder.add_edge("planning_agent", "writing_agent")
graphbuilder.add_edge("writing_agent", "review_agent")
graphbuilder.add_edge("review_agent", "citation_insertor")
graphbuilder.add_edge("citation_insertor", "compile_essay")
graphbuilder.add_edge("compile_essay", "naturalizer_feedback_generator")
graphbuilder.add_edge("naturalizer_feedback_generator", "naturalizer_editor")
graphbuilder.add_edge("naturalizer_editor", "tone_polisher")
graphbuilder.add_edge("tone_polisher", "redundancy_checker")

graph = graphbuilder.compile()

user_input = """
Should ambition be restrained?
"""
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