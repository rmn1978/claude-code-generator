# claude-code-generator
A Python tool that solves Claude 3.7's code generation limits by incrementally building large projects. It breaks code generation into multiple API calls, automatically creates files/directories on your system, maintains conversation context, and tracks progress—allowing for development of projects that exceed Claude's standard response limits.

--How to Use It

1-Install the required packages:

pip install anthropic

2-Set up your API key (either as an environment variable or pass it as an argument):

export ANTHROPIC_API_KEY=your_api_key_here

3-Run the program:

python claude_code_generator.py --output-dir your_project_directory

4-Enter your initial prompt describing the code project you want to build

5-Review and approve each generation round:

The program will show you what Claude has generated and written to disk.
You can continue with the default prompt, modify it, or stop generation.

--Key Features

-Intelligently parses code blocks and file paths from Claude's responses
-Creates necessary directory structures automatically
-Tracks file registry to provide context for subsequent generations
-Allows interactive modification of prompts between generation rounds
-Robust error handling for file operations
-Configurable parameters like model, temperature, and token limits

This solves the problem of Claude stopping after generating large amounts of code by managing the conversation flow and creating files incrementally as they're generated.

:)
