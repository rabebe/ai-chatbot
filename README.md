# Self-Correcting Summarization Agent

This project implements a LangGraph workflow for high-quality, reliable document summarization. The core innovation is a Self-Correction Loop where a Large Language Model (LLM) acts as a Judge to critique the summary draft and send it back for refinement until it meets a high standard of accuracy and completeness.
The application is run via a Command Line Interface (CLI) and uses the Gemini API for all LLM functions.

## Features

- Iterative Refinement: Employs a closed-loop system using LangGraph to iteratively improve the summary draft based on structured feedback.
- LLM-as-a-Judge: Uses a dedicated LLM with a Pydantic schema (JudgeResult) to score and critique the summary, ensuring reliable quality control.
- Structured Output: Relies on structured output generation for predictable, robust state management.
- Modular Architecture: Clear separation of concerns between core state, worker nodes, and graph orchestration.

## Architecture (The Self-Correction Loop)

The process runs through a defined sequence of nodes, managed by LangGraph:
1. Summarizer Node: Generates the initial draft.
2. Judge Node: Evaluates the draft against the original document chunks.
3. Conditional Edge: If the Judge scores the draft below a threshold (or flags a major error), the workflow proceeds to Refine. Otherwise, it proceeds to END.
4. Refinement Node: Takes the Judge's specific critique and revises the summary draft.
5. Loop: The revised summary goes back to the Judge Node for re-evaluation.
This loop repeats until the summary is approved or the maximum refinement steps are reached.

## Setup and Installation

Prerequisites

1. Python 3.10+

2. Gemini API Key: You must have an API key set as an environment variable.

Step-by-Step Installation

1. Clone the Repository:

```
git clone [your_repo_url]
cd self-correcting-summarizer
```

2. Create a Virtual Environment:

```
python -m venv venv
source venv/bin/activate
```

3. Install Dependencies: (Assuming dependencies are listed in requirements.txt)

```
pip install -r requirements.txt
```

(Minimum dependencies include langgraph, langchain-google-genai, python-dotenv, pydantic, etc.)

4. Set Environment Variable:
Create a file named .env in the root directory and add your key:

```
# .env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

## Usage

Run the agent from the root directory using the run_agent.py script. It requires an input file path.

Command Format

```
python run_agent.py <INPUT_FILE_PATH> [--max_steps <INTEGER>]
```

Example

To summarize a document named test_document.txt with a maximum of 3 refinement steps:

```
python run_agent.py data/test_document.txt --max_steps 3
```

The script will log the execution of each node, and the final output will display the approved summary, the final Judge Score, and the total number of refinement steps taken.
