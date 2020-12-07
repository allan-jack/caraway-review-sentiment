#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:06:58 2020

@author: jack
"""

# %%%% Preliminaries and library loading
import datetime
import os
import pandas as pd
import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# import numpy as np

os.chdir('/Users/jack/Documents/xavi_scraping/')
path_to_driver_1 = '/Users/jack/Documents/xavi_scraping/chromedriver_86'

driver = webdriver.Chrome(executable_path=path_to_driver_1)

# %%%% Link
link_to_scrape = 'https://www.carawayhome.com/pages/reviews'
driver.get(link_to_scrape)

# %%%% Gets total number of reviews
# waits for element to be visible
WebDriverWait(driver, 10).until(
    EC.visibility_of_element_located((By.XPATH, "//a[@class='text-m']")))
total_n_reviews_raw = driver.find_element_by_xpath("//a[@class='text-m']").text
total_n_reviews_str = re.findall(r'\d+', total_n_reviews_raw) 
total_n_reviews_list = list(map(int, total_n_reviews_str))
total_n_reviews_int = total_n_reviews_list[0]

# %%%% Scraping
reviews_dict = []

# pull reviews until break
while True:
    
    # waits for the reviews to be visible before grabbing their elements
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, "//div[@class='yotpo-review yotpo-regular-box yotpo-regular-box-filters-padding ']")))
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, "//div[@class='yotpo-review yotpo-regular-box  ']")))
        
    # the first review on each page was different than the other 9, so I am getting two different elements
    reviews = driver.find_elements_by_xpath("//div[@class='yotpo-review yotpo-regular-box yotpo-regular-box-filters-padding ']") + driver.find_elements_by_xpath("//div[@class='yotpo-review yotpo-regular-box  ']")
    
    r = 0
    
    # for loop to scrape through reviews
    for r in range(len(reviews)):
        one_review                     = {}
        one_review['scrapping_date']   = datetime.datetime.now()
        try:                
            soup                       = BeautifulSoup(reviews[r].get_attribute('innerHTML'))
        except:
            # I got StaleElementReferenceException a lot here, so in order to solve this problem,
            # I just refresh the page, make sure the elements are there and reinitialize reviews.
            WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.XPATH, "//div[@class='yotpo-review yotpo-regular-box yotpo-regular-box-filters-padding ']")))
            WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.XPATH, "//div[@class='yotpo-review yotpo-regular-box  ']")))
            reviews = driver.find_elements_by_xpath("//div[@class='yotpo-review yotpo-regular-box yotpo-regular-box-filters-padding ']") + driver.find_elements_by_xpath("//div[@class='yotpo-review yotpo-regular-box  ']")
            soup                       = BeautifulSoup(reviews[r].get_attribute('innerHTML'))
            
        # scrapes raw HTML
        try:
            one_review_raw             = reviews[r].get_attribute('innerHTML')
        except:
            one_review_raw             = ""
        one_review['review_raw']       = one_review_raw
        
        # pulls date from innerHTML soup
        try:
            one_review_date            = soup.find('span', attrs={'class':'y-label yotpo-review-date'}).text
        except:
            one_review_date            = ""
        one_review['review_date']      = one_review_date
        
        # pulls text from innerHTML soup
        try:
            one_review_text            = soup.find('div', attrs={'class':'yotpo-review-wrapper'}).text 
        except:
            one_review_text            = ""
        one_review['one_review_text_raw']  = one_review_text
        
        # pulls product category from innerHTML soup
        try:
            one_review_prod            = soup.find('div', attrs={'class':'yotpo-grouping-reference'}).text 
        except:
            one_review_prod            = ""
        one_review['one_review_prod']  = one_review_prod
        
        # pulls star rating from innerHTML soup
        try:
            one_review_stars           = re.findall('[0-5].[0-9] star rating',reviews[r].get_attribute('innerHTML'))[0]
        except:
            one_review_stars = ""
        one_review['one_review_stars'] = one_review_stars
        
        reviews_dict.append(one_review)
        
    # Keeps clicking next page until all reviews are collected
    if len(reviews_dict) < total_n_reviews_int:
        # clicks for next page
        try:
            # waits for the next page button to be visible, then clicks
            WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.XPATH, "//a[@class='yotpo-page-element yotpo-icon yotpo-icon-right-arrow yotpo_next ']")))
            driver.find_element_by_xpath("//a[@class='yotpo-page-element yotpo-icon yotpo-icon-right-arrow yotpo_next ']").click()
            # time.sleep(2)
        except:
            # exits out of pop up, pop up only occurs if user puts cursor on browser
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//div[@class='email-capture__popup-close popup-close js-offer-close ac-offer-close']"))).click()
            # then clicks next page
            driver.find_element_by_xpath("//a[@class='yotpo-page-element yotpo-icon yotpo-icon-right-arrow yotpo_next ']").click()
            # time.sleep(2)
            
        time.sleep(10)
        # time.sleep(np.random.uniform(10.0, 11.0))
    
    # When all reviews are in the dictionary, the while(True) loop breaks
    else:
        break        

driver.close()

# %%%% Cleaning
caraway_reviews = pd.DataFrame.from_dict(reviews_dict)
caraway_reviews['review_raw'].map(lambda caraway_reviews: BeautifulSoup(caraway_reviews).text)
caraway_reviews['n_stars'] = caraway_reviews.one_review_stars.str.extract('([0-5])(.)(0)( )(star rating)')[[0]]
caraway_reviews['review_text_actual'] = caraway_reviews.one_review_text_raw.str.extract('((?<=stating).*$)')
caraway_reviews['one_review_prod'] = caraway_reviews.one_review_prod.str.extract('((?<=Reviewed on: ).*$)')

# %%%% To excel!
file_name = 'caraway_reviews_dec7.xlsx'
caraway_reviews.to_excel(file_name)