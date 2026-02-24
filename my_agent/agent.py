from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import asyncio


moderator = LlmAgent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='The moderator for this debate.',
    instruction="""You control the debate.
    Current transcript: {transcript}
    Last checked response: {last_checked}
    Decide: Continue (say whose turn) OR End debate and give summary + winner.
    Keep responses neutral and short.""",
)

debator1 = LlmAgent(
    model='gemini-2.5-flash',
    name='positive_debator',
    description='The debator supporting the topic presented.',
    instruction="""You are Debater Pro. Always stay in character.
    Topic: {topic}
    Previous transcript: {transcript}
    Respond concisely (150-250 words).""",
)

debator2 = LlmAgent(
    model='gemini-2.5-flash',
    name='negative_debator',
    description='The debator opposing the topic presented.',
    instruction="""You are Debater Con. Always stay in character.
    Topic: {topic}
    Previous transcript: {transcript}
    Respond concisely (150-250 words).""",
)

factChecker = LlmAgent(
    model='gemini-2.5-flash',
    name='fact_checker',
    description='Checks any factual information presented by the two presentors.',
    instruction="""Review the LAST response only.
    List every factual claim and mark as:
    - ✅ Accurate
    - ⚠️ Partially accurate (explain)
    - ❌ False (correct it with source if possible)
    Give an overall accuracy score 1-10.
    Output in clear bullet format.""",
)

pipeline = SequentialAgent(
    name='debate_pipeline',
    description='',
    sub_agents=[]
)

debateLoop = LoopAgent(
    name='debate_loop',
    description='Loops through the debate.',
    sub_agents=[],
)


