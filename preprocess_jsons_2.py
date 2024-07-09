import os
import pandas as pd
import json
import re
from bs4 import BeautifulSoup
final_div="show-more-less-html__markup relative overflow-hidden show-more-less-html__markup--clamp-after-5"
import json

def load_json(file_path):
    """
    Load a JSON file and return its content as a dictionary.

    :param file_path: str, path to the JSON file.
    :return: dict, content of the JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_div_content_with_regex(html_content):
    """
    Extracts the content inside a div with a class starting with 'show-more-less-html', removing all HTML tags.

    :param html_content: str, the full HTML content.
    :return: str, the plain text content inside the target div.
    """
    # Define the regular expression to find the div with the target class
    pattern = re.compile(r'<div class="show-more-less-html[^"]*">(.*?)</div>', re.DOTALL)

    # Search for the target div
    match = pattern.search(html_content)

    if match:
        # Extract the content inside the div
        div_content = match.group(0)
        div_content=div_content.replace('<br><br>',' ')
        # Remove all HTML tags
        text_content = re.sub(r'<[^>]+>', '', div_content)
        text_content = re.sub(r'"', '', text_content)
        text_content = re.sub(r"'", '', text_content)

        return text_content.strip()
    else:
        return None


def extract_headers_and_emails(html_content):
    """
    Extracts the content inside <h1>, <h4> headers, and any email addresses.

    :param html_content: str, the full HTML content.
    :return: dict, containing headers and emails.
    """
    # Extract <h1> headers
    h1_pattern = re.compile(r'<h1[^>]*>(.*?)</h1>', re.DOTALL)
    h1_headers = [re.sub(r'<[^>]+>', '', match).strip() for match in h1_pattern.findall(html_content)]

    # Extract <h4> headers
    h4_pattern = re.compile(r'<h4[^>]*>(.*?)</h4>', re.DOTALL)
    h4_headers = [re.sub(r'<[^>]+>', '', match).strip() for match in h4_pattern.findall(html_content)]

    email_pattern = re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b',re.DOTALL)
    # Extract email addresses

    emails = email_pattern.findall(html_content)

    return {
        'h1_headers': h1_headers[0],
        'h4_headers': [ re.sub(r'\s+', ' ',h4_head.replace("\n"," ")) for h4_head in h4_headers][1],
        'emails': list(set(emails))
    }

k=0
for j in os.listdir('./raw_data_parsed'):
    file=load_json(os.path.join('./raw_data_parsed',j))
    for i in range(len(file['Company_name'])):
        k=k+1
        try:
            content=extract_div_content_with_regex(file['Job_details'][i])
            rs=extract_headers_and_emails(file['Job_details'][i])
            rs.update({'job_description':content})
            # Convert the dictionary to a JSON object
            if file['Job_details'][i]:
                with open(f'./processed_data/Jop_Application_{k}.json', 'w') as f:
                    json.dump(rs, f)
        except:
            pass