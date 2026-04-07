# MultiAgentDebate

This project facilitates a multi-agent debate about something.

## Running the program

Ensure a .env file is setup containing the following information:
<code>
GOOGLE_GENAI_USE_VERTEXAI=TRUE<br>
GOOGLE_CLOUD_PROJECT=   # Your project id<br>
GOOGLE_CLOUD_LOCATION=  # The region you wish to host the project in<br>
GOOGLE_API_KEY=         # Your Vertex/Gemini API key</code>

Command line:

`adk run my_agent`

Web interface:

`adk web`

When the program is initially run, send a prompt such as 'Begin' or 'Start the debate' to trigger the LoopAgent

## Stopping Conditions

The LoopAgent has a default stopping point of 5 complete rounds for each debater.

The moderator can stop the debate early if it feels that a sufficient end has been reached.

If the user wishes to stop the debate, simply type 'stop' and the moderator should end the debate.

## Manually Debate the Agents

If the user wishes to manually debate one of the agents, then simply change one of the following constants to 'OFF'
```
PRO = 'ON'
CON = 'ON'
```
Whichever side is chosen, when it is that debater's turn, it will wait for user input and continue with the rest of the debate as if the user's response was the output from an agent. The user's response is not checked by the fact-checker.