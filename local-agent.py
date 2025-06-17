import os
import sys

# import ai library 

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "granite3.3:8b"  # You can change this to your preferred model

import requests

# import the client object from openai and pass it my apikey 

# retrieve the contents of the memory.txt file 

def get_memory():
    with open("memory.txt", "r") as f:
        return f.read()
    
def get_summarized_memory(): 
    with open("summarized_txt", "r") as f: 
        return f.read()
    
# function to ask LLM, define the model 

def ask_llm(prompt, model=DEFAULT_MODEL, memory=None):

    # Create the system prompt
    system_message = "You are a helpful research assistant. Always respond in 3 sentences or less. Keep the conversation concise and chatty. Here are the user's profile characteristics: "
    if memory:
        system_message += f" You have the ability to see the memory of the conversation. Here is the memory: {memory}."

    # Ollama expects a single prompt, so combine system and user prompt
    full_prompt = f"{system_message}\nUser: {prompt}"

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": full_prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    result = response.json()
    return result.get("response", "").strip()

# saves the memory of each interaction to a file 

def save_to_memory(prompt, response, model="gpt-4o-mini"):
    with open("memory.txt", "a") as f:
        #it is formatted, as ai: text, new line, and then user: text, new line 
        f.write(f"Model: {model}\nUser: {prompt}\nAI: {response}\n\n")

# summarizes the memory of each interaction in a single sentence to optimize token usage 

import asyncio

def summarize_memory(prompt, model=DEFAULT_MODEL):
    # System prompt for summarization with a few-shot example
    system_message = """You are chatting with a user. If a specific interaction is extremely interesting, 
    you should summarize it in a way that is easy to understand and remember, in a single sentence. 
    The goal is to minimize your token usage. 
    
    Here's an example of how you should save the memory: 

    Interaction 1: the user said that he enjoys programming and I responded with 3 ways to get better.
    Interaction 2: the user mentioned he likes to read books about systems design and I recommended he take a course at MIT. 
    
    """

    # Combine system and user prompt for Ollama
    full_prompt = f"{system_message}\nUser: {prompt}"

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": full_prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    result = response.json()
    summary = result.get("response", "").strip()

    # Write the summarized text to the memory file
    with open("summarized_memory.txt", "a") as f:
        f.write(f"Model: {model}\nSummary: {summary}\n\n")

    return summary


# tool functions with basic implementations

def sentiment_analyzer(text):

    # Use Ollama to analyze sentiment
    system_message = (
        "You are a sentiment analyzer. Analyze the emotional tone of the text and respond with: "
        "POSITIVE, NEGATIVE, or NEUTRAL, followed by a brief explanation."
    )
    full_prompt = f"{system_message}\nAnalyze the sentiment of this text: {text}"

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": DEFAULT_MODEL,
            "prompt": full_prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    result = response.json()
    return result.get("response", "").strip()

def writing_improver(text):
    """Improve the given text using Ollama"""
    system_message = (
        "You are a writing assistant. Improve the given text by making it clearer, more engaging, and better structured. "
        "Return only the improved version."
    )
    full_prompt = f"{system_message}\nImprove this text: {text}"

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": DEFAULT_MODEL,
            "prompt": full_prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    result = response.json()
    return result.get("response", "").strip()

def researcher_agent(topic):
    """Research the given topic"""
    system_message = "You are a research assistant. Provide helpful, accurate information about the given topic in 2-3 sentences."
    full_prompt = f"{system_message}\nResearch and provide information about: {topic}"

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": DEFAULT_MODEL,
            "prompt": full_prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    result = response.json()
    return result.get("response", "").strip()

# A tool is designed to parse a response. 
def native_parser(interaction):
    """Parse the interaction and select the appropriate tool using Ollama"""
    import json

    system_message = """You are a helpful assistant that selects the appropriate tool based on user input.

You have these tools available:
- Sentiment Analyzer: Analyzes the emotional tone of text
- Writing Improver: Helps improve and refine written content  
- Researcher Agent: Researches topics and provides information

You must respond with ONLY a JSON object in this exact format:
{
  "tool": "tool_name_here",
  "arguments": "the_text_to_process"
}

Examples:
- If user asks about emotions: {"tool": "Sentiment Analyzer", "arguments": "the text to analyze"}
- If user wants writing help: {"tool": "Writing Improver", "arguments": "the text to improve"}  
- If user asks questions: {"tool": "Researcher Agent", "arguments": "the topic to research"}

Return ONLY the JSON, no other text."""

    full_prompt = f"{system_message}\nThe user said: {interaction}"

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": DEFAULT_MODEL,
            "prompt": full_prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    response_content = response.json().get("response", "").strip()

    try:
        parsed_response = json.loads(response_content)
        tool_name = parsed_response.get("tool")
        arguments = parsed_response.get("arguments")
        return tool_name, arguments
    except json.JSONDecodeError:
        print(f"Error: Could not parse response as JSON: {response_content}")
        return None, None

# a tool executor that takes the function that was called and executes it 

def tool_executor(tool_selection, arguments):
    """Execute the appropriate tool"""
    if not tool_selection or not arguments:
        return "Error: Invalid tool selection or arguments."
        
    if tool_selection == "Sentiment Analyzer":
        return sentiment_analyzer(arguments)
    elif tool_selection == "Writing Improver":
        return writing_improver(arguments)
    elif tool_selection == "Researcher Agent":
        return researcher_agent(arguments)
    else:
        return f"Error: Unknown tool '{tool_selection}'. Available tools: Sentiment Analyzer, Writing Improver, Researcher Agent."


# Agent scratchpad to track all interactions and reasoning
class AgentScratchpad:
    def __init__(self):

        # This will give the scratchpad a few places to save different types of data in a list format 

        # interaction var
        self.interactions = []
        # current plan var 
        self.current_plan = []
        # completed steps var  
        self.completed_steps = []
        
    # adds an interaction to the scratchpad 
    def add_interaction(self, step_type, content, result=None):
        """Add an interaction to the scratchpad"""

        # an interaction is composed of the step, the type of step, the content, and the result 
        interaction = {
            "step": len(self.interactions) + 1,
            "type": step_type,  # "observation", "thought", "action", "final_answer"
            "content": content,
            "result": result
        }
        # the result is appended to the interaction variable as a dictionary (key-value pairs)
        self.interactions.append(interaction)
        
    def get_scratchpad_summary(self):
        """Get a formatted summary of all interactions"""
        summary = "SCRATCHPAD:\n"
        for interaction in self.interactions:
            summary += f"Step {interaction['step']} - {interaction['type'].upper()}: {interaction['content']}\n"
            if interaction['result']:
                summary += f"RESULT: {interaction['result']}\n"
            summary += "\n"
        return summary
        
    def clear(self):
        """Clear the scratchpad for new conversation"""
        self.interactions = []
        self.current_plan = []
        self.completed_steps = []

# Enhanced parser that can handle planning and final answer detection


# The parser will always look at how the final answer is being produced. We are essentially creating completions. 

def agent_parser(user_input, scratchpad):
    """Parse user input and determine next action using scratchpad context (Ollama version)"""
    import json

    scratchpad_context = scratchpad.get_scratchpad_summary()

    system_prompt = """You are an intelligent agent that can plan and execute tasks using tools.

You have these tools available:
- Sentiment Analyzer: Analyzes emotional tone of text
- Writing Improver: Improves and refines written content  
- Researcher Agent: Researches topics and provides information

Your response must be a JSON object with ONE of these formats:

"Respond ONLY with a valid JSON object as specified above. Do not include any text, comments, or explanations before or after the JSON."

FOR PLANNING (when you need to think about approach):
{
  "action_type": "plan",
  "thought": "your reasoning about what needs to be done",
  "plan": ["step 1", "step 2", "step 3"]
}

FOR USING A TOOL:
{
  "action_type": "tool",
  "tool": "tool_name_here",
  "arguments": "text_to_process",
  "reasoning": "why you're using this tool"
}

FOR FINAL ANSWER (when you have enough information):
{
  "action_type": "final_answer",
  "answer": "your complete answer to the user"
}

"Respond ONLY with a valid JSON object as specified above. Do not include any text, comments, or explanations before or after the JSON."



Look at the scratchpad to see what's already been done. Don't repeat actions unnecessarily."""

    full_prompt = (
        f"{system_prompt}\n\n"
        f"User request: {user_input}\n\n"
        f"Current scratchpad:\n{scratchpad_context}"
    )

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": DEFAULT_MODEL,
            "prompt": full_prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    response_content = response.json().get("response", "").strip()

    try:
        parsed_response = json.loads(response_content)
        return parsed_response
    except json.JSONDecodeError as e:
        print(f"Error parsing agent response: {e}")
        print(f"Raw response: {response_content}")
        return {"action_type": "error", "message": "Failed to parse agent response"}

# Agent execution loop
def run_agent(user_input, scratchpad, max_steps=10):
    """Run the agent loop until final answer or max steps reached"""
    
    # Adds an interaction to the scratchpad with the observation 
    scratchpad.add_interaction("observation", f"User request: {user_input}")
    
    for step in range(max_steps):
        print(f"\n--- Agent Step {step + 1} ---")
        
        # Get next action from the scratchpad
        next_action = agent_parser(user_input, scratchpad)


        # 
        if next_action["action_type"] == "plan":
            # Agent is planning
            thought = next_action.get("thought", "")
            plan = next_action.get("plan", [])
            
            scratchpad.add_interaction("thought", thought)
            scratchpad.add_interaction("plan", f"Plan: {'; '.join(plan)}")
            
            print(f"🤔 THINKING: {thought}")
            print(f"📋 PLAN: {'; '.join(plan)}")
            
        elif next_action["action_type"] == "tool":
            # Agent wants to use a tool
            tool_name = next_action.get("tool")
            arguments = next_action.get("arguments")
            reasoning = next_action.get("reasoning", "")
            
            print(f"🔧 USING TOOL: {tool_name}")
            print(f"💭 REASONING: {reasoning}")
            
            # Execute the tool
            tool_result = tool_executor(tool_name, arguments)
            
            scratchpad.add_interaction("action", f"Used {tool_name} with: {arguments}", tool_result)
            
            print(f"📊 RESULT: {tool_result}")
            
        elif next_action["action_type"] == "final_answer":
            # Agent has final answer
            final_answer = next_action.get("answer", "")
            
            scratchpad.add_interaction("final_answer", final_answer)
            
            print(f"✅ FINAL ANSWER: {final_answer}")
            return final_answer
            
        elif next_action["action_type"] == "error":
            error_msg = next_action.get("message", "Unknown error")
            print(f"❌ ERROR: {error_msg}")
            return f"Sorry, I encountered an error: {error_msg}"
            
        else:
            print(f"❓ UNKNOWN ACTION: {next_action}")
            
    # If we reach max steps without final answer
    return "I've reached my thinking limit. Let me try to give you the best answer I can based on what I've learned so far."

if __name__ == "__main__":

    # Create the agent scratchpad
    scratchpad = AgentScratchpad()
    
    # set the main loop to true so that it can run the program 
    main_app_loop = True 

    # print the welcome message 
    print("AI: Welcome to your intelligent CLI agent! I can analyze sentiment, improve writing, and research topics.")
    print("AI: I'll show you my thinking process as I work through your requests.")

    # start the main loop 
    while main_app_loop: 
        # take in user input
        user_input = input("\nYou: ")

        # if the user types exit, he can leave the chat interface 
        if user_input.lower() == "exit":
            print("See you later! Come back again!")
            main_app_loop = False 
        else:
            # Clear scratchpad for new conversation
            scratchpad.clear()
            
            # Run the agent with the user input
            final_answer = run_agent(user_input, scratchpad)
            
            # Print the final answer
            print(f"\n🎯 AI: {final_answer}")
            
            # Save this interaction to memory (optional - you can keep this for long-term memory)
            save_to_memory(user_input, final_answer)
            
            # Summarize memory (optional - you can keep this for long-term memory)
            memory_to_summarize = f"User: {user_input}\nAI: {final_answer}"
            summarize_memory(memory_to_summarize)
