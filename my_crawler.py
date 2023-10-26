import queue
import re
import time
from urllib.parse import urljoin

import matplotlib.pyplot as plt
import networkx as nx
import requests
from bs4 import BeautifulSoup
from networkx.drawing.nx_pydot import write_dot

max_depth = 2
s_no = 1
starting_link = "https://www2.ed.gov/about/offices/list/ocr/docs/tix_dis.html"
to_be_crawled = []
crawled_urls = []
next_urls = queue.Queue()

match_patterns = ['U.S. Department', 'Title IX', 'Commission', 'regulations', 'Sex', 'Rights', 'Discrimination', 'Law']

for i in range(0, len(match_patterns)):
    match_patterns.append(match_patterns[i].lower())

def ismatch_title_link_map(s):
    for i in match_patterns:
        if re.search(i, s):
            return True
    return False

def add_data(text_data):
    text_list = text_data.split("\n")
    data = ''
    for text in text_list:
        if len(text) > 1 and not text.isspace():
            data = data + '\n' + text.strip()
    bytedata = data.encode('utf-8')
    binary_file = open("{}.txt".format(s_no), "ab")
    binary_file.write(bytedata)
    binary_file.close()
    return bytedata

def get_data_from_url(url):
    child_link = []
    try:
        html_page = requests.get(url).text
        soup = BeautifulSoup(html_page, 'html.parser')
        text_data = soup.text.strip()
        bytedata = add_data(text_data)
        sublinks = soup.find_all("a")
        # print("Sublink - ", len(sublinks))
        for sublink in sublinks:
            if ismatch_title_link_map(sublink.text.strip()):
                if sublink.get("href") and sublink["href"][0] == '/':
                    path = urljoin(url, sublink["href"])
                    child_link.append(path)
                elif sublink.get("href") and sublink["href"][:4] == 'http':
                    path = sublink["href"]
                    child_link.append(path)
        return bytedata, child_link
    except Exception as e:
        print("Exception - ", e)
    # print("Child Link - ", len(child_link))
    

def find_links(url_with_depth_tuple, G):
    (url, current_depth) = url_with_depth_tuple
    print("Current Depth - ", current_depth)
    if current_depth < max_depth:
        bytedata, child_link = get_data_from_url(url)
        G.add_node(url)
        # G.nodes[url]['data'] = bytedata
        G.nodes[url]['child_link'] = child_link
        # print("Added Parent Node")
        for link in child_link:
            child_bytedata, child_children_link = get_data_from_url(link)
            G.add_node(link)
            # G.nodes[link]['data'] = child_bytedata
            G.nodes[link]['child_link'] = child_children_link
            G.add_edge(url, link)
            # print("added edge")
            if link not in crawled_urls:
                next_urls.put((link, current_depth + 1))
                crawled_urls.append(link)
        # print("Added Child Node")

def crawl_data(next_urls, G):
    while not next_urls.empty():
        find_links(next_urls.get(), G)


start_time = time.time()
next_urls.put((starting_link, 0))
crawled_urls.append(starting_link)
G = nx.Graph()
crawl_data(next_urls, G)
nx.draw(G, with_labels = True)
plt.savefig("1.png")
write_dot(G, '1.dot')
end_time = time.time()
print("Total Time - {}".format(end_time - start_time))