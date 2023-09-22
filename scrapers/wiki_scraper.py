
import os
import csv
import urllib.request
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
from zipfile import ZipFile
from os import path
import json
import requests
import codecs
from bs4.element import Comment
import glob
import argparse

parser = argparse.ArgumentParser(description='Scrape wikitravel.')
parser.add_argument('-q', '--query',type=str,required=True,help="The search term to look for.")
parser.add_argument('-l', '--level',type=float,default=1,help="Depth of scraping every link.")
parser.add_argument('-r','--request_id',type=str,default=1_1,help="Name of the destination directory for the downloaded content.")
args = parser.parse_args()

query = args.query
level = args.level
request_id = args.request_id

html_directory = "/home/kk/wikitravel/static/"+request_id+"/"
paragraph_text_directory = "/home/kk/wikitravel/static/"+request_id+"/paragraph_text_files"

def get_wikitravel_link(query):    
    
    '''
    Takes a Wiki travel query (String) and returns the first url (String) given by wikitravel
    Tested and works get_wikitravel_link('hampi') -> /en/hampi
    '''
    print('Function running : get_wikitravel_link')
    print('User Query is: ' + query)
    query = query.replace(" ", "+")     # In case query has more than one word
    pattern = '[0-9]'                   # In case the user query has a number
    query = re.sub(pattern, '', query)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1)' }  # I'm a fucking pirate
    url = 'https://wikitravel.org/wiki/en/index.php?search=' + query + '&title=Special%3ASearch&profile=default&fulltext=1'
    req = Request(url=url, headers=headers) 
    html = urlopen(req).read()   
    # print(html) 

    # Getting only the top result after parsing the response
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.findAll(attrs={'class':'mw-search-result-heading'}):  
        children = link.findChildren("a" , recursive=False)        
        for child in children:            
            result = child['href']
            break
        break
    # print(result)
    return result


def get_links_from_wikitravel_page(url,level,request_id):
    '''
    Takes a url of a wiki-travel page and returns a list of urls in the page worth scraping further
    '''
    print('Function running : get_links_from_wikitravel_page')
    # print('Main page url is: ' + url)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1)' }  # I'm a fucking pirate
    req = Request(url=url, headers=headers)
    html = urlopen(req).read()    
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    # Getting only the links found in the Body, and some dirty regex hack
    for body in soup.findAll(attrs={'class':'mw-body-content', 'id':'bodyContent'}):
        for link in body.find_all("a",{'class':''}, href=re.compile("^/en/((?!File:|Wikitravel:|Category:|Special:).)*$")):
            url = link['href']
            link_processor_and_page_downloader(url,request_id)
            main_page_url = 'https://www.wikitravel.org'+url

            if level > 1:
                links.extend(get_links_from_wikitravel_page(main_page_url,level-1,request_id))
            else:
                # print("url: ",url)
                links.append(url)
    return links

def link_processor_and_page_downloader(url,request_id):
    wget_url = url.replace("(", "\(")
    wget_url = wget_url.replace(")", "\)")
    command = 'wget -q -N -c -k -p -e robots=off -U mozilla -K -E -t 6 --no-check-certificate --span-hosts --convert-links --no-directories --directory-prefix=static/'+ request_id +' https://www.wikitravel.org' + wget_url
    os.system(command)

def wikitravel_scraper(query, request_id, level):
    '''
    takes as input a query, scraps and saves it in the response folder with title as request id
    tested and works - wikitravel_scraper('Hampi', '1_1', 0)
    TO DO - change the main page html to redirect to the lower level htmls
    '''
    url = get_wikitravel_link(query) 
    link_processor_and_page_downloader(url,request_id)
    if level >= 1:
        main_page_url = 'https://www.wikitravel.org'+url
        links = get_links_from_wikitravel_page(main_page_url,level,request_id)
        html_to_text()
    else:
        print("front page downloaded........................")


def html_to_text():
    if not os.path.exists(paragraph_text_directory):
        os.mkdir(paragraph_text_directory)
    files = glob.glob("*.html",root_dir=html_directory)
    
    for file in files:
        path = os.path.join(html_directory,file)
        with codecs.open(path, 'r', encoding="utf8") as f:
            soup = BeautifulSoup(f, 'html.parser')
            paragraph_content = soup.find_all('p')
            paragraph_content =  "".join(t.text.strip().replace(".",".\n").lstrip() for t in paragraph_content)
            paragraph_text_path = os.path.join(paragraph_text_directory,file.split(".html")[0]+".txt")
            with open(paragraph_text_path,"w") as t:
                t.write(paragraph_content)

wikitravel_scraper(query, request_id, level)
