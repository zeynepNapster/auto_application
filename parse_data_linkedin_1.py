from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
import json


# Function to extract job details
def extract_job_details(driver, job_links):
    job_details = []
    for job_link in job_links:
        try:
            driver.get(job_link)
            time.sleep(3)
            job_details.append(driver.page_source)
        except Exception as e:
            print(f"Error extracting job details: {e}")
            continue
    return job_details


# Function to extract job data from a link
def extract_job_data(driver, link):
    driver.get(link)
    time.sleep(3)

    try:
        job_count = int(driver.find_element(By.CLASS_NAME, 'results-context-header__job-count').text.replace(',', '').replace('+', ''))
    except Exception as e:
        job_count=1000
    CT=0
    for _ in range((job_count + 24) // 25):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

        load_more_button = driver.find_element(By.XPATH, "//button[@aria-label='See more jobs']")
        driver.execute_script("arguments[0].click();", load_more_button)
        time.sleep(3)
        CT=CT+1
        if CT>200:
            break


    company_names = [elem.text for elem in driver.find_elements(By.CLASS_NAME, 'base-search-card__subtitle')]
    title_names = [elem.text for elem in driver.find_elements(By.CLASS_NAME, 'base-search-card__title')]
    job_links = [elem.get_attribute('href') for elem in driver.find_elements(By.CSS_SELECTOR, 'a.base-card__full-link')]

    job_details = extract_job_details(driver, job_links)
    print(len(company_names))
    return {
        'Company_name': company_names,
        'Title_name': title_names,
        'Job_details': job_details,
        'URL':job_links
    }


# Main script
links=['https://www.linkedin.com/jobs/search?keywords=Electrical%20Engineering&location=Munich&geoId=100477049&distance=25&f_E=2&f_TPR=&f_WT=1%2C3%2C2&position=1&pageNum=0',
       'https://www.linkedin.com/jobs/search?keywords=technical%20porcurement&location=Munich&geoId=100477049&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0',
'https://www.linkedin.com/jobs/search?keywords=Power%20Engineering&location=Munich&geoId=100477049&distance=25&f_TPR=&f_E=2&position=1&pageNum=0'
,'https://www.linkedin.com/jobs/search?keywords=Solar%20Engineer&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0'
,'https://www.linkedin.com/jobs/search?keywords=Solar%20Energy%20Engineering&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0'
,'https://www.linkedin.com/jobs/search?keywords=Renewable%20Energy%20Engineer&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&distance=25&f_TPR=&f_E=2&position=1&pageNum=0'
'https://www.linkedin.com/jobs/search?keywords=Engineer%20Electrical%20Systems&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&distance=25&f_TPR=&f_E=2&position=1&pageNum=0',
'https://www.linkedin.com/jobs/search?keywords=Engineering&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&distance=25&f_E=2&f_PP=100477049&f_TPR=&f_WT=3%2C2&position=1&pageNum=0'
'https://www.linkedin.com/jobs/search?keywords=Industrial%20Engineering&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0',
    'https://www.linkedin.com/jobs/search?keywords=Industrial%20Engineering&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&distance=25&f_TPR=&f_E=2&position=1&pageNum=0',
 'https://www.linkedin.com/jobs/search?keywords=Electrical%20Engineer&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&distance=25&f_TPR=&f_E=2&position=1&pageNum=0',
'https://www.linkedin.com/jobs/search?keywords=Project%20Management&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&distance=25&f_TPR=&f_E=2&position=1&pageNum=0',
    'https://www.linkedin.com/jobs/search?keywords=Electronics%20Engineering&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0',
    'https://www.linkedin.com/jobs/search?keywords=Power%20Engineering&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&distance=25&f_TPR=&f_E=2&position=1&pageNum=0',
    'https://www.linkedin.com/jobs/search?keywords=Mechanical%20Engineer&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&distance=25&f_TPR=&f_E=2&position=1&pageNum=0',
'https://www.linkedin.com/jobs/search?keywords=Engineer&location=M%C3%BCnchen%2C%20Bayern%2C%20Deutschland&geoId=100477049&distance=25&f_TPR=&f_PP=100495942&position=1&pageNum=0',
    'https://www.linkedin.com/jobs/search?keywords=electrical%20Engineer&location=Augsburg%2C%20Bayern%2C%20Deutschland&geoId=103849782&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0',
    'https://www.linkedin.com/jobs/search?keywords=Project%20Engineer&location=Augsburg%2C%20Bayern%2C%20Deutschland&geoId=103849782&distance=25&f_TPR=&f_E=2&position=1&pageNum=0',
'https://www.linkedin.com/jobs/search?keywords=power%20engineer&location=Augsburg%2C%20Bayern%2C%20Deutschland&geoId=103849782&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0']

options = webdriver.ChromeOptions()
options.add_experimental_option('prefs', {'intl.accept_languages': 'en,en_US'})
driver = webdriver.Chrome(options=options)
driver.implicitly_wait(10)

for idx, link in enumerate(links):
    try:
        job_data = extract_job_data(driver, link)


        if job_data:
            with open(f'./raw_data_parsed/linked_in_{idx + 1}.json', 'w') as outfile:
                json.dump(job_data, outfile)
    except Exception as e:
        print(f"Error processing link {link}: {e}")

driver.quit()
