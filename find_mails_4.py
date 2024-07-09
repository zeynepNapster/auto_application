from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
import pandas as pd
import os
import re
from langchain import PromptTemplate

df=pd.read_csv('archieve/Expo_Companies.csv').rename(columns={'Unnamed: 0': 'Companies'})
df.dropna(subset=['Locations'],inplace=True)
df=df[df['Locations'].str.match('.*(munich|MUNICH|Munich).*')]
os.environ["OPENAI_API_KEY"] = 'sk-proj-cUi8xZ3zfWwyQmhKqQMjT3BlbkFJKS5AevM7ONvEF8930fXQ'

wikipedia = WikipediaAPIWrapper()
def filter_alphabetic(content):
    """
    Filters out everything but alphabetic characters from the given content.

    :param content: str, the input text.
    :return: str, the text with only alphabetic characters.
    """
    return re.sub(r'[^a-zA-Z\s]', '', content)
from langchain.agents import Tool
search = DuckDuckGoSearchRun()
tools = [
    Tool(
    name='wikipedia',
    func=wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia"
),
 Tool(
    name='DuckDuckGo Search',
    func= search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)

]


from langchain.agents import initialize_agent
from langchain import OpenAI

llm = OpenAI(temperature=0,model_name='gpt-4')
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=100,  # Increase this value for more iterations
    max_execution_time=600,
reduce_k_below_max_tokens=True,
    max_tokens = 4400,
)

prompt_template: str = """
You are a Web Research Assistant. Your only goal is to find an email related to the following company: {company} and return it.
First try to find the email on the companies website and its impressum otherwise use other resources like linkedin or xing and finally return 
the email adress
"""
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(prompt_template)
import json
from pypdf import PdfReader
def load_json(file_path):
    """
    Load a JSON file and return its content as a dictionary.

    :param file_path: str, path to the JSON file.
    :return: dict, content of the JSON file.
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

for k,linked_in_card in enumerate(os.listdir('./processed_data')):#



    try:
        file = load_json(os.path.join('./processed_data', linked_in_card))
        name = filter_alphabetic(file['h1_headers'])
        if os.path.exists(os.path.join(f'./mails_2/{name}_mail.txt')):
            continue

        res=zero_shot_agent.run(prompt.format(company=file['h4_headers']))

        with open(os.path.join(f'./mails_2/{name}_mail.txt'), 'w', encoding='utf-8') as file:
            file.write(str(res))

    except:
        pass

