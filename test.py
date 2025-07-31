import openai
from crewai import Agent, Task, Crew
 
# ✅ Step 1: Configure your Azure OpenAI client
client = openai.AzureOpenAI(
    api_key="sk-hMERVV5GYatUo2VYTLZn3Q",
    api_version="2023-06-01-preview",
    azure_endpoint="https://genai-sharedservice-emea.pwcinternal.com"
)
 
# ✅ Step 2: Create a wrapper to make it compatible with CrewAI
class AzureOpenAIWrapper:
    def __init__(self, client, model="azure.gpt-4o", temperature=0.3):
        self.client = client
        self.model = model
        self.temperature = temperature
 
    def complete(self, prompt, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=500
        )
        return response.choices[0].message.content
 
    def __call__(self, prompt, **kwargs):
        return self.complete(prompt, **kwargs)
 
    # 👇 This is what CrewAI uses internally
    def call(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=500
        )
        return {
            "output": response.choices[0].message.content
        }
 
 
# ✅ Step 3: Initialize the wrapper
llm = AzureOpenAIWrapper(client)
 
# ✅ Step 4: Create the agent
info_agent = Agent(
    role="Information Agent",
    goal="Provide compelling information about a certain topic.",
    backstory="""
        You're a trivia enthusiast with a passion for facts.
        Known for dominating pub quizzes, your friends rely on you for quick, reliable knowledge.
    """,
    verbose=True,
    llm=llm
)
 
# ✅ Step 5: Define the task
task1 = Task(
    description="Tell me about the blue-ringed octopus.",
    expected_output="Provide a quick summary followed by 7 bullet points describing it.",
    agent=info_agent
)
 
# ✅ Step 6: Create and run the crew
crew = Crew(
    agents=[info_agent],
    tasks=[task1],
    verbose=True
)
 
result = crew.kickoff()
 
# ✅ Step 7: Print the result
print("\n######################")
print(result)
 
 
 
import openai
from crewai import Agent, Task, Crew
from crewai.llm import BaseLLM
from typing import List, Dict
 
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
 
# ✅ Step 4: Create the agent
info_agent = Agent(
    role="Information Agent",
    goal="Provide compelling information about a certain topic.",
    backstory="""
        You're a trivia enthusiast with a passion for facts.
        Known for dominating pub quizzes, your friends rely on you for quick, reliable knowledge.
    """,
    verbose=True,
    llm=llm
)
 
# ✅ Step 5: Define the task
task1 = Task(
    description="Tell me about the blue-ringed octopus.",
    expected_output="Provide a quick summary followed by 7 bullet points describing it.",
    agent=info_agent
)
 
# ✅ Step 6: Create and run the crew
crew = Crew(
    agents=[info_agent],
    tasks=[task1],
)
 
result = crew.kickoff()
 
# ✅ Step 7: Print the result
print("\n######################")
print(result)
 
 
 