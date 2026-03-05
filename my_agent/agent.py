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

# Constants
APP_NAME = "MultiAgentDebate"
USER_ID = "12345"
SESSION_ID = "123344"
MODEL = "gemini-2.5-flash"
TOPIC = "Is AI better than human intelligence?"

# Custom Debate Agent
from google.adk.agents import LlmAgent, BaseAgent

class DebateAgent(BaseAgent):
    """
    Custom agent for a debate.
    This agent orchestrates the debate, handling ordering and loops.
    """

    def __init__(
        self,
        name: str,
        moderator: LlmAgent,
        pro: LlmAgent,
        con: LlmAgent,
        fact_checker: LlmAgent,
    ):
        moderator= moderator
        pro=pro
        con=con
        fact_checker= fact_checker

        super().__init__(
            name=name,
            sub_agents=[moderator, pro, con, fact_checker],
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Implements the custom debate logic.
        """
        print(f"[{self.name}] Starting debate.")

        state = ctx.session.state
        state.setdefault("topic", TOPIC)
        state.setdefault("transcript", "")
        state.setdefault("last_checked", "")
        state.setdefault("current_speaker", "")

        iterations = 0
        max_iterations = 10

        async def async_input(prompt: str) -> str:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                return await loop.run_in_executor(executor, input, prompt)

        flag = False
        round = 1
        while iterations < max_iterations and not flag:
            iterations += 1
            round = iterations/2 if iterations % 2 == 0 else round
            if iterations % 2 == 1:
                print(f"\n[{self.name}] --- Round {iterations} ---\n")

            # Moderator
            print(f"[{self.name}] Running Moderator...")
            async for event in moderator.run_async(ctx):
                yield event

            mod_decision = state.get("mod_decision", "").strip()
            if mod_decision.upper().startswith("END:"):
                flag = True

            if flag == False:
                if iterations > 1 and not mod_decision.upper().startswith("NEXT:"):
                    # default alternation
                    next_speaker = "Con" if state["current_speaker"] == "Pro" else "Pro"
                else:
                    next_speaker = mod_decision.split("NEXT:", 1)[1].strip()
                
                state["current_speaker"] = next_speaker

                # Run debater
                print(f"\n[{self.name}] Running {next_speaker}...")
                debater = con if next_speaker  == "Con" else pro
                async for event in debater.run_async(ctx):
                    yield event

                # Fact check
                print(f"\n[{self.name}] Running Fact Checker...")
                async for event in fact_checker.run_async(ctx):
                    yield event

                # Transcript update (add safety)
                last_resp = state.get("last_response", "[no response]")
                last_chk  = state.get("last_checked", "[no check]")
                state["transcript"] += f"{next_speaker}: {last_resp}\nFact check:\n{last_chk}\n\n"

                print("\n")

        print(f"\n[{self.name}] Debate finished.")

# Define the LLM agents
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
- If transcript is empty OR user says anything like "start", "begin", "go", "debate", "hello", or 
just presses send, introduce the topic in 2-3 sentences, then output NEXT: Pro
- If you feel the debate has reached it's conclusion, either by repetitiveness, lack of new content, 
derailed arguments, or something else, then output "END: <reason>"
- Otherwise, evaluate the last round and decide:
  - "NEXT: Pro"
  - "NEXT: Con"

Output ONLY one line: "NEXT: Pro", "NEXT: Con" or "END: <reason>"
At the beginning of all of your responses, output "[Moderator]: <message>"
""",
    output_key="mod_decision",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1
    )
)

con = LlmAgent(
    name="Con",
    model=MODEL,
    instruction="""
You are debating as the opposing debater.
You argue AGAINST the topic passionately and logically.
You present facts and statistics to back your claims.
No matter what the topic is, you oppose it completely.
Topic: {topic}
Transcript so far: {transcript}
Opponent's last checked response: {last_checked}
Argue clearly, logically, passionately. Keep response concise (100-150 words).
At the beginning of all of your responses, output "[Con]: <argument>"
""",
    output_key="last_response",
    generate_content_config=types.GenerateContentConfig(
        temperature=1.2
    )
)

pro = LlmAgent(
    name="Pro",
    model=MODEL,
    instruction="""
You are debating as the supporting debater.
You argue FOR the topic passionately and logically.
You present facts and statistics to back your claims.
No matter what the topic is, you support it completely.
Topic: {topic}
Transcript so far: {transcript}
Opponent's last checked response: {last_checked}
Argue clearly, logically, passionately. Keep response concise (100-150 words).
At the beginning of all of your responses, output "[Pro]: <argument>"
""",
    output_key="last_response",
    generate_content_config=types.GenerateContentConfig(
        temperature=1.2
    )
)

fact_checker = LlmAgent(
    model=MODEL,
    name='fact_checker',
    description='You fact check every single claim rigorously.',
    instruction="""
The topic of the debate: {topic}
Review the LAST response only: {last_response}
List every factual claim and mark as:
✅ Accurate
⚠️ Partially accurate (explain BRIEFLY)
❌ False (correct it with source if possible)
Give an overall accuracy score 1-10.
Output in clear bullet format.
Do not exceed 150 words in your output.
Be concise but strict.
At the beginning of all of your responses, output "[Fact Checker]: <message>"
""",
    output_key="last_checked",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2
    )
)

INITIAL_STATE = {"topic": TOPIC}

# Setup Runner and Session
async def setup_session_and_runner():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID, 
        state=INITIAL_STATE.copy(),
    )
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    return session_service, runner

# Function to Interact with the Agent
async def run_debate():
    session_service, runner = await setup_session_and_runner()
    session = session_service.sessions[APP_NAME][USER_ID][SESSION_ID]
    session.state["topic"] = TOPIC

    print(f"\nStarting debate on: {TOPIC}\n")

    # Required: give the agent something to react to
    start_message = types.Content(
        role="user",
        parts=[
            types.Part(
                text="Start the debate on the topic in state now."
            )
        ]
    )

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=start_message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)
        elif hasattr(event, 'type'):
            print("Error event:", event)
        else:
            print("Event:", event)  # debug all events

    print("\nFinished.")
    
# Create the custom agent instance
root_agent = DebateAgent(
    name="DebateAgent",
    moderator=moderator,
    pro=pro,
    con=con,
    fact_checker=fact_checker
)

# For standalone script
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_debate())