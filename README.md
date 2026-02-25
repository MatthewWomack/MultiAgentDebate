# MultiAgentDebate

This project facilitates a multi-agent debate about something.

## Running the program

Ensure a .env file is setup containing the following information:

<code>GOOGLE_GENAI_USE_VERTEXAI=TRUE<br>
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

If the user wishes to stop it, then oh well.
