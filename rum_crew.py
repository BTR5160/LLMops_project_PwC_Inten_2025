import openai
from crewai import Agent, Task, Crew
from textwrap import dedent

import openai
from crewai import Agent, Task, Crew
from crewai.llm import BaseLLM
from typing import List, Dict 

# === Import functional pipeline logic ===
from pipeline.loader import load_files
from pipeline.extractor import extract_content
from pipeline.preprocess import preprocess_text
from pipeline.relevance import is_relevant
from pipeline.judge import judge_chunk
from pipeline.storage import save_chunk
import os

# CrewAI-compatible LLM
# ✅ Step 1: Configure your Azure OpenAI client
client = openai.AzureOpenAI(
    api_key="sk-hMERVV5GYatUo2VYTLZn3Q",
    api_version="2023-06-01-preview",
    azure_endpoint="https://genai-sharedservice-emea.pwcinternal.com"
)
 
# ✅ Step 2: Define a CrewAI-compatible LLM wrapper
class AzureOpenAIWrapper(BaseLLM):
    def __init__(self, client, model="azure.gpt-4o", temperature=0.3):
        self.client = client
        self.model = model
        self.temperature = temperature
 
    def call(self, messages: List[Dict], **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
 
    def complete(self, prompt: str, **kwargs) -> str:
        return self.call([{"role": "user", "content": prompt}])
 
# ✅ Step 3: Initialize the wrapper
llm = AzureOpenAIWrapper(client)
 


# === Agents ===
loader_agent = Agent(
    role="File Loader",
    goal="Load files from disk and provide metadata and content",
    backstory="You are responsible for handling all types of documents and preparing them for extraction.",
    verbose=True,
    llm=llm
)

extractor_agent = Agent(
    role="Content Extractor",
    goal="Extract meaningful text from various file types including PDFs, DOCX, images, and audio.",
    backstory="You are skilled in OCR, transcription, and AI-based fallback techniques.",
    verbose=True,
    llm=llm
)

preprocessor_agent = Agent(
    role="Text Preprocessor",
    goal="Clean, chunk, and deduplicate raw extracted text for further evaluation.",
    backstory="You are a data cleaner and transformer who prepares content for LLM consumption.",
    verbose=True,
    llm=llm
)

relevance_agent = Agent(
    role="Finance Relevance Evaluator",
    goal="Evaluate whether a text chunk is relevant to the finance domain.",
    backstory="You are a domain expert in financial data analysis and model fine-tuning.",
    verbose=True,
    llm=llm
)

judge_agent = Agent(
    role="Judgment Agent",
    goal="Make the final decision to fine-tune, vectorize, or discard a chunk based on quality.",
    backstory="You are a senior AI filter responsible for curating the final dataset.",
    verbose=True,
    llm=llm
)

# === Tasks ===
loader_task = Task(
    description=dedent("""
        Load files from the data/input directory. Handle PDFs, DOCX, images, audio, and text files.
        Return a structured dictionary with file name, extension, and raw content.
    """),
    expected_output="A list of dictionaries, each with 'name', 'type', 'content'",
    agent=loader_agent,
    async_execution=False,
    function=lambda: load_files("data/input")
)

extractor_task = Task(
    description=dedent("""
        For each file loaded, extract useful textual content using OCR, Whisper, or LLM-based captioning.
        Handle edge cases like charts or handwritten text. Ensure output is clean and text-based.
    """),
    expected_output="Clean plain text from each file",
    agent=extractor_agent,
    async_execution=False,
    function=lambda files: [extract_content(f) for f in files]
)

preprocessor_task = Task(
    description=dedent("""
        Process the extracted content. Clean noise, chunk into 500-800 tokens, and remove duplicates.
    """),
    expected_output="List of unique, clean text chunks",
    agent=preprocessor_agent,
    async_execution=False,
    function=lambda texts: [preprocess_text(t) for t in texts]
)

relevance_task = Task(
    description=dedent("""
        Review each chunk and answer whether it is finance-related and useful for training a financial model.
    """),
    expected_output="Yes or No for each chunk",
    agent=relevance_agent,
    async_execution=False,
    function=lambda chunk_lists: [[chunk for chunk in chunks if is_relevant(chunk)] for chunks in chunk_lists]
)

judge_task = Task(
    description=dedent("""
        For chunks marked relevant, classify them as:
        - fine_tune: high-quality, structured data
        - vector_only: useful but not training-grade
        - discard: not usable
    """),
    expected_output="One of: fine_tune, vector_only, discard",
    agent=judge_agent,
    async_execution=False,
    function=lambda relevant_chunks_list: [save_chunk(chunk, judge_chunk(chunk), "data/output") for chunks in relevant_chunks_list for chunk in chunks]
)

# === Crew Execution ===
crew = Crew(
    agents=[loader_agent, extractor_agent, preprocessor_agent, relevance_agent, judge_agent],
    tasks=[loader_task, extractor_task, preprocessor_task, relevance_task, judge_task],
    verbose=True
)

result = crew.kickoff()
print("\n🎯 Final result:", result)

