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
import configs
wikipedia = WikipediaAPIWrapper()
search = DuckDuckGoSearchRun()
py=PythonREPL()

yf=YahooFinanceNewsTool()
search_2=SerpAPIWrapper(serpapi_api_key=configs.SERPAPI_API_KEY)
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
pdf_file=PdfReader('cv/ResumeeZeynepTozge.pdf')
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
pdf_file=PdfReader('cv/ResumeeZeynepTozge.pdf')
page = pdf_file.pages[0]


import openai


for k,linked_in_card in enumerate(os.listdir('./processed_data/jsons')):#



    try:
        file = load_json(os.path.join('./processed_data/jsons', linked_in_card))
        name = filter_alphabetic(file['h1_headers'])
        if os.path.exists(os.path.join(f'./applications/{name}_application.txt')):
            continue

        prompt_template: str = """
### Instructions:

As an expert in crafting professional application emails, your task is to generate an engaging and well-structured job application email for the position listed on LinkedIn. 

Below are the details you need to incorporate:

- Job Title: {job}
- Job Description: {jobdescription}
- Company Name: {company}

### Context:

1. Identify and elaborate on key symbiotic relationships between the job description and the applicant's CV, showcasing why the candidate is an excellent fit for the role.
2. If the job description is in German, reply in English, emphasizing the candidate's commitment to improving their German skills through intensive courses.
3. Exclude mention of academic degrees if the job does not specifically require them, as the applicant already holds a master’s degree.
4. Ignore internships and working student positions if they are not relevant to the application.

### Desired Outcome:

Create a professional and compelling email that highlights the candidate's qualifications, experience, and enthusiasm for the role, making it clear why they would be a valuable addition to the company.

### Email Format and Style:

- Professional and concise language
- Well-structured paragraphs
- A friendly but formal tone
- Length: Around 3-4 short paragraphs

### Example:

Subject: Application for {job} at {company}

Dear Hiring Committee,

I am writing to express my interest in the {job} position at {company}, as advertised. With a strong background in {relevant experience/skills}, I am confident that my expertise aligns perfectly with the requirements of this role.

In my previous role at {Previous Company}, I effectively {relevant achievement/task}, which mirrors the responsibilities outlined in your job description. Additionally, I am currently enhancing my German language skills through extensive courses to ensure seamless communication within your team and with clients, if necessary.

I am excited about the opportunity to contribute to {company} and am particularly drawn to {specific aspect of the company or job}. I am eager to bring my {specific skills or attributes} to your team and help achieve continued success.

Thank you for considering my application. I look forward to the possibility of discussing my application further.

Best regards,
[Your Name]

---
        """
        prompt = PromptTemplate.from_template(prompt_template)
        pp=prompt.format(job=file['h1_headers'], company=file['h4_headers'],
                              jobdescription=file['job_description'])



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