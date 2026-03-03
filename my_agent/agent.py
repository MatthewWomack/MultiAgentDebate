'''TO DO
1) Add human-in-the-loop interaction before the start of each round
2) Change endDebateTool to be cleaner
3) Store past debates and be able to access them

Optional: Add voices

'''
#from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent, callback_context
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools import BaseTool
from typing import Optional

TOPIC = "Does pineapple belong on pizza?" # CHANGE BEFORE DUE DATE

class endDebateTool(BaseTool):
    name = "end_debate"
    description = "Call this ONLY when you are certain the debate should end now. Provide a short reason."

    async def run(self, reason: str, ctx):  # ctx is passed automatically
        ctx.event_actions.escalate = True  # This breaks the LoopAgent immediately
        return f"Debate ended early: {reason}"

moderator = LlmAgent(
    model='gemini-2.5-flash',
    name='Moderator',
    description='The moderator for this debate.',
    instruction="""You are a strict, neutral moderator who is moderating a 
    debate in front of a crowd.
    Current full transcript:{transcript}

    Latest fact-checked round:{last_checked}

    Topic:{topic}

    Your tasks:
    1. Introduce the debate if the transcript is empty. Briefly (2-3 sentences) introduce the 
    topic and then begin the debate.
    2. Decide if the round is valid (no major off-topic/toxicity/stalling).
    3. If the debate should continue output "NEXT: Pro" or "NEXT: Con"
    4. If it's time to end (e.g., clear winner, both sides exhausted, enough rounds, stalemate, 
    etc.) Call 'endDebateTool' with a 1-sentence reason. Output the message and call the tool.

    Output your decision OR tool call — nothing else.""",
    tools=[endDebateTool],
    output_key="mod_decision",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2
    )
)

from google.adk.tools import FunctionTool

def get_human_argument(argument: str):
    """
    Submits the human's response to the debate.
    Args:
        argument: Your actual debate argument text.
    """
    return argument

def is_human_turn(ctx, *args, **kwargs) -> bool:
    state = ctx.session.state
    # If mode is HUMAN and speaker is 'Con', ADK Web will pop up the input box
    return state.get("mode") == "AI_VS_HUMAN" and state.get("current_speaker") == "Con"

current_debater = LlmAgent(
    name="Current_Debater",
    model='gemini-2.0-flash', # Ensure you are using a stable 2.0 version
    instruction="""You are representing: {current_speaker}.
    
    CRITICAL:
    - If MODE is 'AI_VS_HUMAN' and speaker is 'Con', call 'get_human_argument'.
    - Otherwise (AI_VS_AI mode), generate a 150-word argument for {topic}.""",
    tools=[
        # This parameter is the magic that triggers the ADK Web UI pause
        FunctionTool(get_human_argument, require_confirmation=is_human_turn)
    ],
    output_key="last_response"
)

factChecker = LlmAgent(
    model='gemini-2.5-flash',
    name='fact_checker',
    description='You fact check every single claim rigorously.',
    instruction="""Review the LAST response only.
    List every factual claim and mark as:
    - ✅ Accurate
    - ⚠️ Partially accurate (explain BRIEFLY)
    - ❌ False (correct it with source if possible)
    Give an overall accuracy score 1-10.
    Output in clear bullet format.""",
    output_key="last_checked",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2
    )
)

debate_round= SequentialAgent(
    name="Debate_Round",
    sub_agents=[moderator, current_debater, factChecker]
)

root_agent = LoopAgent(
    name='debate_loop',
    max_iterations=10,
    description='Loops through alternating debate rounds.',
    sub_agents=[debate_round] # Start with the pro round
)

def ensure_defaults(callback_context: callback_context.CallbackContext) -> Optional[types.Content]:
    state = callback_context.state
    state.setdefault("topic", TOPIC)
    state.setdefault("current_speaker", "Pro") 
    state.setdefault("transcript", "")
    state.setdefault("mode", "AI_VS_HUMAN") 
    state.setdefault("last_response", "")
    state.setdefault("last_checked", "")
    state.setdefault("moderator_decision", "")

root_agent.before_agent_callback = ensure_defaults

def alternateSpeaker(context):
    current= context.session.state.get("current_speaker", "Pro")

    next_speaker= "Con" if current == "Pro" else "Pro"

    context.session.state["current_speaker"] = next_speaker

    last_resp = context.session.state.get("last_response", "")
    last_check = context.session.state.get("last_checked", "")
    mod_dec = context.session.state.get("mod_decision", "")

    if last_resp:
        context.session.state["transcript"] += f"\n\n{current}: {last_resp}\nFact-check: {last_check}\nModerator: {mod_dec}"

    return None

root_agent.after_agent_callback= alternateSpeaker

async def debate():
    service = InMemorySessionService()
    session = await service.create_session(
        app_name="MultiAgentDebate",
        user_id="user1",
        session_id="debate1"
    )

    runner = Runner(agent=root_agent, session_service=service)
    message = types.Content(
        role= "user",
        parts=[types.Part.from_text(text="Begin the debate now.")]
    )
    runner.run(user_id="user1", session_id="debate1", new_message=message)

    updated_session = service.get_session(
        app_name="MultiAgentDebate",
        user_id="user1",
        session_id="debate1"
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(debate())