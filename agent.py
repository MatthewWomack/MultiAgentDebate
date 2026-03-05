"""
DESCRIPTION: 
    Implements a multi-agent debate system using the Google ADK. 
    The system uses a 'DebateAgent' as a root orchestrator to manage 
    a Moderator, two opposing Debaters, and a Fact Checker.
    
    Supports:
    - AI vs AI debates.
    - Human vs AI debates (via state-based pausing).
    - Factual verification of every turn.
    - Persistent session state management via InMemorySessionService.

@author : Donnell Wilkins
@author : Matthew Womack
@author : Prashreet Poudel

Date: 3/5/2026
"""

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

# Constants
APP_NAME = "MultiAgentDebate"
USER_ID = "12345"
SESSION_ID = "123344"
MODEL = "gemini-2.5-flash"
TOPIC = "Is the war with Iran worth it?"
PRO = "ON"
CON = 'ON'

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Debate Agent
from google.adk.agents import LlmAgent, BaseAgent


class DebateAgent(BaseAgent):
    """
    Custom agent for orchestrating a structured debate between two opposing sides.
    
    This agent manages the complete debate flow by coordinating multiple sub-agents:
    - Moderator: Controls turn-taking and debate progression
    - Pro: Argues in favor of the topic (can be AI or human)
    - Con: Argues against the topic (can be AI or human)
    - Fact Checker: Verifies factual claims in each response
    
    The agent supports both AI-vs-AI and Human-vs-AI debate formats through
    configurable state flags. It maintains debate state including transcript,
    current speaker, and fact-checking results.
    
    Attributes:
        name (str): Name of the debate agent
        moderator (LlmAgent): Agent responsible for moderating the debate
        pro (LlmAgent): Agent arguing for the topic
        con (LlmAgent): Agent arguing against the topic
        fact_checker (LlmAgent): Agent that verifies factual claims
    """

    def __init__(
        self,
        name: str,
        moderator: LlmAgent,
        pro: LlmAgent,
        con: LlmAgent,
        fact_checker: LlmAgent,
    ):
        """
        Initialize the DebateAgent with its constituent sub-agents.
        
        Args:
            name (str): Unique identifier for this debate agent
            moderator (LlmAgent): The moderator agent that controls debate flow
            pro (LlmAgent): The agent arguing for the proposition
            con (LlmAgent): The agent arguing against the proposition
            fact_checker (LlmAgent): The agent that verifies factual claims
        """
        moderator = moderator
        pro = pro
        con = con
        fact_checker = fact_checker

        super().__init__(
            name=name,
            sub_agents=[moderator, pro, con, fact_checker],
        )
        
        
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Implements the core debate orchestration logic with human fact-checking.
        """
        logger.info(f"[{self.name}] Starting debate.")

        state = ctx.session.state
        state.setdefault("topic", TOPIC)
        state.setdefault("transcript", "")
        state.setdefault("last_checked", "")
        state.setdefault("last_response", "")
        state.setdefault("current_speaker", "Pro")
        state.setdefault("pro_on", PRO)
        state.setdefault("con_on", CON)

        iterations = 0
        max_iterations = 10
        flag = False

        while iterations < max_iterations and not flag:
            iterations += 1
            logger.info(f"\n\n[{self.name}] --- Round {iterations} ---")

            # 1. Moderator decides who speaks next
            logger.info(f"[{self.name}] Running Moderator...")
            async for event in moderator.run_async(ctx):
                yield event

            mod_decision = state.get("mod_decision", "").strip()
            print(f"Moderator: {mod_decision!r}")

            if mod_decision.upper().startswith("END:"):
                logger.info(f"[{self.name}] Debate ended: {mod_decision}")
                flag = True
                break

            # 2. Determine next speaker
            if iterations > 1 and not mod_decision.upper().startswith("NEXT:"):
                current = state.get("current_speaker", "Pro")
                next_speaker = "Con" if current == "Pro" else "Pro"
            else:
                try:
                    next_speaker = mod_decision.split(":", 1)[1].strip().title()
                except IndexError:
                    next_speaker = "Pro"

            state["current_speaker"] = next_speaker
            logger.info(f"[{self.name}] Next speaker: {next_speaker}")

            # 3. Run the debater (AI or human) with proper human-vs-AI handling
            human_paused = False

            is_human = (next_speaker == "Pro" and state.get("pro_on", "OFF").upper() == "OFF") or \
                       (next_speaker == "Con" and state.get("con_on", "OFF").upper() == "OFF")

            if is_human:
                # Pause if human has not yet provided input
                if not state.get("last_response"):
                    logger.info(f"[{self.name}] Waiting for human {next_speaker} input in Web UI...")
                    human_paused = True
            else:
                # Run AI agent
                agent = pro if next_speaker == "Pro" else con
                async for event in agent.run_async(ctx):
                    if event.content and event.content.parts:
                        state["last_response"] = event.content.parts[0].text
                    yield event

            if human_paused:
                return  # pause until human input

            # 4. Run fact checker on last_response (human or AI)
            logger.info(f"[{self.name}] Running Fact Checker on last response...")
            # Mark whether the input was from human or AI for clarity
            state["last_response_to_check"] = ("[Human input] " if is_human else "[AI input] ") + state.get("last_response", "")

            async for event in fact_checker.run_async(ctx):
                if event.content and event.content.parts:
                    state["last_checked"] = event.content.parts[0].text
                yield event

            # 5. Update transcript safely
            state["transcript"] += f"{next_speaker}: {state.get('last_response', '[no response]')}\n" \
                                   f"Fact check:\n{state.get('last_checked', '[no check]')}\n\n"

            # 6. Clear last_response for next round
            state["last_response"] = ""

        logger.info(f"\n[{self.name}] Debate finished.")
        
        
        

# Define the LLM agents
moderator = LlmAgent(
    model=MODEL,
    name="Moderator",
    description="The moderator for this debate.",
    instruction="""
        You are a neutral and fair debate moderator.

        Debate Topic:
        {topic}

        Debate configuration:
        Pro side: {pro_on}  (ON = AI debating, OFF = Human debating)
        Con side: {con_on}  (ON = AI debating, OFF = Human debating)

        Current speaker:
        {current_speaker}

        Full transcript so far:
        {transcript}

        Latest fact-check:
        {last_checked}

        Your job:
        Maintain a structured debate and decide who speaks next.

        Rules:

        1. If the transcript is empty OR the user says things like
        "start", "begin", "go", "debate", "hello", or sends an empty message,
        briefly introduce the topic in 1–2 sentences and then output:

        NEXT: Pro

        2. After each round:
        - Review the last argument
        - Consider the fact-check results
        - Decide which side should respond next

        3. Alternate speakers naturally unless the debate flow suggests
        one side should respond again.

        4. If the debate becomes repetitive, unproductive, or clearly reaches
        a conclusion, end it with:

        END: <short reason>

        Important:
        You DO NOT control whether the speaker is AI or human.
        Your job is ONLY to choose which side speaks next.

        Output EXACTLY one line in one of these formats:

        NEXT: Pro
        NEXT: Con
        END: <reason>
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
        Topic: {topic}
        Transcript so far: {transcript}
        Opponent's last checked response: {last_checked}
        Mode: if {con_on} is on then:
            You are debating as the opposing debater.
            You argue AGAINST the topic passionately and logically.
            You present facts and statistics to back your claims.
            Argue clearly, logically, passionately. Keep response concise (100-150 words).
            Output only your argument, no extra commentary.
        If it is not on let the user write the Con argument.
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
    Topic: {topic}
    Transcript so far: {transcript}
    Opponent's last checked response: {last_checked}
    Mode: if {pro_on} is on then:
        You are debating as the supporting debater.
        You argue FOR the topic passionately and logically.
        You present facts and statistics to back your claims.
        Argue clearly, logically, passionately. Keep response concise (100-150 words).
        Output only your argument, no extra commentary.
    if it is not on let the user type the argument""",
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
Output in clear bullet format.""",
    output_key="last_checked",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2
    )
)

INITIAL_STATE = {"topic": TOPIC}

# Setup Runner and Session
async def setup_session_and_runner():
    """
    Initialize and configure the session service and runner for the debate system.
    
    This function creates an in-memory session service, establishes a new session
    with the initial debate state, and sets up a runner to execute the debate agent.
    
    Returns:
        tuple: A tuple containing:
            - session_service (InMemorySessionService): The configured session service
            - runner (Runner): The configured runner for executing the debate agent
            
    Example:
        >>> session_service, runner = await setup_session_and_runner()
        >>> # Use the runner to start the debate
    """
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID, 
        state=INITIAL_STATE.copy(),
    )
    logger.info(f"Initial session state: {session.state}")
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    return session_service, runner

# Function to Interact with the Agent
async def run_debate():
    """
    Execute a complete debate session from start to finish.
    
    This function orchestrates the entire debate process:
    1. Sets up the session and runner
    2. Initializes the debate topic
    3. Sends a start message to trigger the debate
    4. Streams and displays all events from the debate
    5. Handles both AI and human responses
    
    The function is designed to be run as a standalone script or called
    programmatically. It uses asynchronous streaming to process events
    in real-time as they are generated by the debate agents.
    
    Returns:
        None: The function prints debate progress and results to stdout
        
    Raises:
        Various exceptions from the underlying ADK framework may be propagated
        
    Example:
        >>> asyncio.run(run_debate())
        Starting debate on: Is AI better than human intelligence?
        [Debate output streams here...]
        Finished.
    """
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