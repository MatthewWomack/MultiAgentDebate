import logging
from typing import AsyncGenerator
from typing_extensions import override
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# -----------------------
# Constants / Configuration
# -----------------------
APP_NAME = "MultiAgentDebate"
USER_ID = "12345"
SESSION_ID = "123344"
MODEL = "gemini-2.5-flash"
TOPIC = "Is AI better than human intelligence?"

# Hard-coded side modes
PRO_MODE = "human"  # "human" or "AI"
CON_MODE = "AI"     # "human" or "AI"

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Custom Debate Agent
# -----------------------
class DebateAgent(BaseAgent):
    """
    Custom agent for a debate.
    This agent orchestrates the debate, handling ordering and loops.
    """

    def __init__(self, name: str, moderator: LlmAgent, pro: LlmAgent, con: LlmAgent, fact_checker: LlmAgent):
        super().__init__(name=name, sub_agents=[moderator, pro, con, fact_checker])

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        # -----------------------
        # Defaults
        # -----------------------
        state.setdefault("topic", TOPIC)
        state.setdefault("transcript", "")
        state.setdefault("last_checked", "")
        state.setdefault("last_response", "")
        state.setdefault("current_speaker", "Pro")
        state.setdefault("mod_decision", "")

        # -----------------------
        # Run Moderator
        # -----------------------
        async for event in self.sub_agents[0].run_async(ctx):
            yield event

        mod_decision = state.get("mod_decision", "").strip()

        if mod_decision.upper().startswith("END:"):
            yield Event.final_response(types.Content(parts=[types.Part(text=mod_decision)]))
            return

        # -----------------------
        # Determine next speaker
        # -----------------------
        if mod_decision.upper().startswith("NEXT:"):
            next_speaker = mod_decision.split(":", 1)[1].strip().title()
        else:
            current = state.get("current_speaker", "Pro")
            next_speaker = "Con" if current == "Pro" else "Pro"

        state["current_speaker"] = next_speaker

        # -----------------------
        # Run Debater with hard-coded modes
        # -----------------------
        if next_speaker == "Pro":
            if PRO_MODE == "human":
                human_input = state.get("human_input", "[No human input]")
                state["last_response"] = human_input
                yield Event.final_response(types.Content(parts=[types.Part(text=human_input)]))
            else:
                async for event in self.sub_agents[1].run_async(ctx):
                    yield event
        else:
            if CON_MODE == "human":
                human_input = state.get("human_input", "[No human input]")
                state["last_response"] = human_input
                yield Event.final_response(types.Content(parts=[types.Part(text=human_input)]))
            else:
                async for event in self.sub_agents[2].run_async(ctx):
                    yield event

        # -----------------------
        # Run Fact Checker
        # -----------------------
        async for event in self.sub_agents[3].run_async(ctx):
            yield event

        # -----------------------
        # Update transcript
        # -----------------------
        last_resp = state.get("last_response", "")
        last_chk = state.get("last_checked", "")
        state["transcript"] += f"{next_speaker}: {last_resp}\nFact check:\n{last_chk}\n\n"
        logger.info(f"[{self.name}] Turn Complete.")


# -----------------------
# Define Agents
# -----------------------
moderator = LlmAgent(
    model=MODEL,
    name='Moderator',
    description='The moderator for this debate.',
    instruction="""
You are a neutral, fair debate moderator.
Current topic: {topic}

Full transcript so far:
{transcript}

Latest fact-check:
{last_checked}

Rules:
- Introduce the topic in 2-3 sentences if the transcript is empty.
- Decide which speaker goes next: output ONLY "NEXT: Pro" or "NEXT: Con".
- If debate is over, output "END: <reason>".
""",
    output_key="mod_decision",
    generate_content_config=types.GenerateContentConfig(temperature=0.1)
)

con = LlmAgent(
    name="Con",
    model=MODEL,
    instruction="""
You are debating as the opposing debater (Con).
Topic: {topic}
Transcript so far: {transcript}
Opponent's last checked response: {last_checked}

Rules:
- If the mode is 'human', respond to the human's last input.
- If the mode is 'AI', generate a full Con argument for AI vs AI.
- Argue logically and passionately, max 150 words.
- Output only your argument.
""",
    output_key="last_response",
    generate_content_config=types.GenerateContentConfig(temperature=1.2)
)

pro = LlmAgent(
    name="Pro",
    model=MODEL,
    instruction="""
You are debating as the supporting debater (Pro).
Topic: {topic}
Transcript so far: {transcript}
Opponent's last checked response: {last_checked}

Rules:
- If the mode is 'human', summarize human input and respond as Pro.
- If the mode is 'AI', generate a full Pro argument for AI vs AI.
- Argue logically and passionately, max 150 words.
- Output only your argument.
""",
    output_key="last_response",
    generate_content_config=types.GenerateContentConfig(temperature=1.2)
)

fact_checker = LlmAgent(
    model=MODEL,
    name='Fact_Checker',
    description='You fact check every single claim rigorously.',
    instruction="""
Topic: {topic}
Review the LAST response only: {last_response}
List every factual claim and mark as:
✅ Accurate
⚠️ Partially accurate
❌ False (correct with source if possible)
Give an overall accuracy score 1-10.
Output in bullet format.
""",
    output_key="last_checked",
    generate_content_config=types.GenerateContentConfig(temperature=0.2)
)

# -----------------------
# Initial State
# -----------------------
INITIAL_STATE = {"topic": TOPIC, "human_input": ""}

# -----------------------
# Runner and Session Setup
# -----------------------
async def setup_session_and_runner():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=INITIAL_STATE.copy(),
    )
    logger.info(f"Initial session state: {session.state}")
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    return session_service, runner

# -----------------------
# Run Debate
# -----------------------
async def run_debate():
    session_service, runner = await setup_session_and_runner()
    session = session_service.sessions[APP_NAME][USER_ID][SESSION_ID]
    session.state["topic"] = TOPIC

    print(f"\nStarting debate on: {TOPIC}\n")

    # Example human input
    session.state["human_input"] = "I believe human intelligence is superior because of creativity and emotional understanding."

    start_message = types.Content(
        role="user",
        parts=[types.Part(text="Start the debate.")]
    )

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=start_message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)

    print("\nDebate finished.")

# -----------------------
# Create Agent Instance
# -----------------------
root_agent = DebateAgent(name="DebateAgent", moderator=moderator, pro=pro, con=con, fact_checker=fact_checker)

# -----------------------
# Standalone Script
# -----------------------
if __name__ == "__main__":
    asyncio.run(run_debate())