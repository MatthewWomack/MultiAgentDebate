from google.adk.agents import LoopAgent, LlmAgent
from google.genai import types

def search() -> str:

    return""

moderator = LlmAgent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='The moderator for this debate.',
    instruction="""You are the moderator for a debate. You must present the topic to the two 
    debators and act professionally in moderating the discussion. You should leave the majority of 
    the speaking to the debators and only speak to progress the debate forward such as allowing 
    each debator to speak and present their claims and prompt them to do so.""",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens=100,
    )
)

debator1 = LlmAgent(
    model='gemini-2.5-flash',
    name='positive_debator',
    description='The debator supporting the topic presented.',
    instruction='',
    tools=[search],
)

debator2 = LlmAgent(
    model='gemini-2.5-flash',
    name='negative_debator',
    description='The debator opposing the topic presented.',
    instruction='',
    tools=[search],
)

factChecker = LlmAgent(
    model='gemini-2.5-flash',
    name='fact_checker',
    description='Checks any factual information presented by the two presentors.',
    instruction='',
    tools=[search],
)

debatePipeline = LoopAgent(
    name='debate_loop',
    description='Loops through the debate.',
    sub_agents=[moderator, debator1, factChecker, moderator, debator2, factChecker],
)