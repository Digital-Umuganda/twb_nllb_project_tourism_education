import csv
import time
from selenium import webdriver
import os

driver = webdriver.Firefox()
directory = "data"
if not os.path.exists(directory):
    os.mkdir(directory)
destinations_to_scrap = [{"path_to_file":"explore_rwanda_tours.csv","link":"https://www.tripadvisor.com/Attraction_Review-g293829-d10725992-Reviews-Explore_RwandaTours-Kigali_Kigali_Province.html","pages_to_scrape":108},{"path_to_file":"nyungwe_national_park","link":"https://www.tripadvisor.com/Attraction_Review-g480231-d479306-Reviews-Nyungwe_National_Park-Butare_Southern_Province.html","pages_to_scrape":233},{"path_to_file":"gorilla_trek_reviews.csv","link":"https://www.tripadvisor.com/Attraction_Review-g293829-d6686686-Reviews-Gorilla_Trek_Africa-Kigali_Kigali_Province.html","pages_to_scrape":647}]
for destination in destinations_to_scrap:
    csvFile = open(os.path.join(directory,destination["path_to_file"]), 'a', encoding="utf-8")
    csvWriter = csv.writer(csvFile)
    driver.get(destination["link"])
    print("--->Scraping the reviews for: ",destination['path_to_file'].split(".csv")[0])
    for i in range(0, destination["pages_to_scrape"]):

        # give the DOM time to load
        time.sleep(6)

        # Click the "expand review" link to reveal the entire review.
        # driver.find_element("xpath",".//span[contains(@class, 'biGQs _P uuBRH')]").click()
        # driver.find_element("xpath",".//span[contains(@class, 'biGQs _P uuBRH')]").click()

        # Now we'll ask Selenium to look for elements in the page and save them to a variable. First lets define a  container that will hold all the reviews on the page. In a moment we'll parse these and save them:
        container = driver.find_elements("xpath","//div[@class='_c']")

        # Next we'll grab the date of the review:
        dates = driver.find_elements("xpath",".//div[@class='RpeCd']")

       # Now we'll look at the reviews in the container and parse them out
        
        for j in range(len(container)): # A loop defined by the number of reviews
# 
            # Grab the title
            # title = container[j].find_element("xpath",".//span[contains(@class='yCeTE')]").text
# 
            # Grab the review
            review = container[j].find_element("xpath",".//div[@class='biGQs _P pZUbB KxBGd']").text.replace("\n","  ")
# 
            # Grab the date
            date = " ".join(dates[j].text.split(" ")[-2:])
# 
            # Save that data in the csv and then continue to process the next review
            csvWriter.writerow([date, review])

        # When all the reviews in the container have been processed, change the page and repeat
        next_page = driver.find_element("xpath",'.//a[@class="BrOJk u j z _F wSSLS tIqAi unMkR"]')
        driver.execute_script("arguments[0].click();",next_page)

driver.quit()
