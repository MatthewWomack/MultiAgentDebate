import asyncio
from typing import Optional
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent, callback_context
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools import BaseTool
'''TO DO
1) Add human-in-the-loop interaction before the start of each round
2) Change endDebateTool to be cleaner
3) Store past debates and be able to access them

Optional: Add voices

'''

TOPIC = "Does pineapple belong on pizza?"

class endDebateTool(BaseTool):
    name = "end_debate"
    description = "Call this to end the debate if the user wants to stop or it's finished."
    async def run(self, reason: str, ctx):
        ctx.event_actions.escalate = True
        return f"Debate concluded: {reason}"



# This agent processes the user's input and generates the AI response

moderator = LlmAgent(
    name='Moderator',
    model='gemini-2.0-flash',
    instruction="""Analyze the conversation. Topic: {topic}.

    1. CHECK FOR COMMANDS: If the user types '/simulate', the mode is now 'AI vs AI'.
    2. END DEBATE: If the human says they are done or wants to stop, call 'end_debate'
    3. ROLE ASSIGNMENT:
       - If mode is 'Human vs AI': Summarize the human's last point.
       - If mode is 'AI vs AI': Act as the PRO-PINEAPPLE debater. Generate a 150-word
         Pro-pineapple argument yourself based on culinary science.
    Output your summary or your Pro-argument for the AI_Opponent to rebut.""",

    tools=[endDebateTool],

    output_key="mod_remark"

)


ai_debater = LlmAgent(
    name="AI_Opponent",
    model='gemini-2.0-flash',
    instruction="""Topic: {topic}. Your Stance: CON (Against).
    Respond to the Human's last point logicially and passionately.
    Stay in character. Keep it under 150 words.""",
    output_key="last_response"

)


fact_checker = LlmAgent(
    name='Fact_Checker',
    model='gemini-2.0-flash',
    instruction="""Review the Human and AI_Opponent's last response.
    Verify claims with ✅, ⚠️, or ❌.""",
    output_key="last_checked"

)
fact_checker = LlmAgent(
    name='Fact_Checker',
    model='gemini-2.0-flash',
    instruction="""You are an elite debate auditor. 
    Review the two distinct arguments made this round:

    1. THE PRO ARGUMENT (Human/Simulated): Found in '{mod_remark}'.
    2. THE CON ARGUMENT (AI Opponent): Found in '{last_response}'.

    FOR EACH SIDE:
    - List factual claims and verify with ✅, ⚠️, or ❌.
    - Identify any logical fallacies (e.g., 'Appeal to Tradition' from the AI or 'Hasty Generalization' from the Human).
    - Give a brief explanation for any flags.""",
    output_key="last_checked"
)
# This is the "Entry Point" the ADK is looking for

root_agent = SequentialAgent(
    name="Debate_Logic",
    sub_agents=[moderator, ai_debater, fact_checker]

)


# Initialize the state

def ensure_defaults(*args, **kwargs):
    # Usually, the context is the first positional argument
    if args:
        ctx = args[0]
    # Sometimes it's passed as a keyword

    else:
        ctx = kwargs.get('callback_context') or kwargs.get('ctx')
    if ctx:
        state = ctx.state
        state.setdefault("topic", TOPIC)
        state.setdefault("transcript", "")
        state.setdefault("last_response", "")
        state.setdefault("last_checked", "")
        state.setdefault("mod_decision", "")



root_agent.before_agent_callback = ensure_defaults