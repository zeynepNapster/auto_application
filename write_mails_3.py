import os
import pandas as pd
import json
import re
from bs4 import BeautifulSoup
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
import pandas as pd
import os
from langchain import PromptTemplate
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.agents import initialize_agent
from langchain.utilities import SerpAPIWrapper
import pandas as pd
import os
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain import PromptTemplate
wikipedia = WikipediaAPIWrapper()
search = DuckDuckGoSearchRun()
py=PythonREPL()
os.environ["OPENAI_API_KEY"] = 'sk-proj-cUi8xZ3zfWwyQmhKqQMjT3BlbkFJKS5AevM7ONvEF8930fXQ'
SERPAPI_API_KEY="1c1954d76a0d8fbe1d0f21eff33b3f213f367a517e4a66cfba078d79cfd4d797"
yf=YahooFinanceNewsTool()
search_2=SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
tools = [
    Tool(
    name='wikipedia',
    func=wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia"
),Tool(name='yahoofinance',func=yf.run,description='useful to get information about financals'),
    Tool(name='pythonrepl',func=py.run,description='useful to use python to calculate stuff'),
 Tool(
    name='DuckDuckGo Search',
    func= search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
),            Tool(
                name="Search",
                func=search_2.run,
                description="""Use this tool to search the web for information"""

            )

]

from langchain import OpenAI
def filter_alphabetic(content):
    """
    Filters out everything but alphabetic characters from the given content.

    :param content: str, the input text.
    :return: str, the text with only alphabetic characters.
    """
    return re.sub(r'[^a-zA-Z\s]', '', content)
llm = OpenAI(temperature=0)
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=100,  # Increase this value for more iterations
    max_execution_time=600
)

from pypdf import PdfReader
pdf_file=PdfReader('ResumeeZeynepTozge.pdf')
page = pdf_file.pages[0]
import pandas as pd
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain import PromptTemplate
def load_json(file_path):
    """
    Load a JSON file and return its content as a dictionary.

    :param file_path: str, path to the JSON file.
    :return: dict, content of the JSON file.
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

prompt_template: str = """
You are an Application Assistant helping a person with the following Curriculum Vitae {CV}.
Write an application email for the following job on LinkedIn {job} described as follows: {jobdescription}
 for this company: {company}. 
 Try to identify symbiotic relationships between the job description 
 and the applicant's CV that would make the candidate a good fit. 
 If the job description is in German, still reply in English, and make sure to highlight
  how the candidate is improving their German skills through extensive courses.
   If the job does not require academic degrees, leave them out. Also, ignore internships and working student offers.
"""
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(prompt_template)
from pypdf import PdfReader
pdf_file=PdfReader('ResumeeZeynepTozge.pdf')
page = pdf_file.pages[0]


import openai


for k,linked_in_card in enumerate(os.listdir('./processed_data')):#



    try:
        file = load_json(os.path.join('./processed_data', linked_in_card))
        name = filter_alphabetic(file['h1_headers'])
        if os.path.exists(os.path.join(f'./applications/{name}_application.txt')):
            continue

        prompt_template: str = """
        Write an application email for the following job on LinkedIn {job} described as follows: {jobdescription}
         for this company: {company}. 
         Try to identify symbiotic relationships between the job description 
         and the applicant's CV that would make the candidate a good fit and elaborate on that.
         If the job description is in German, still reply in English, and make sure to highlight
          how the candidate is improving their German skills through extensive courses.
           If the job does not require academic degrees, leave them out since the applicant already finished its master degree.
           Also, ignore internships and working student offers.
        """
        prompt = PromptTemplate.from_template(prompt_template)
        pp=prompt.format(job=file['h1_headers'], company=file['h4_headers'],
                              jobdescription=file['job_description'])


        print(file['h4_headers'])
        res= openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": f"You are an Application Assistant helping a person to write a application mail given an linkedin job offer and the persons cv {page.extract_text()}"},
                {"role": "user", "content": pp}
            ]
        )


        with open(os.path.join(f'./applications/{name}_application.txt'), 'w', encoding='utf-8') as file:
            file.write(str(res.choices[0].message["content"]))
    except:
        pass