import queue
import re
import time
from urllib.parse import urljoin
# import threading

import matplotlib.pyplot as plt
import networkx as nx
import requests
from bs4 import BeautifulSoup
from networkx.drawing.nx_pydot import write_dot

# max_threads=40
max_depth = 1
match_patterns = ['U.S. Department', 'Title IX', 'Commission', 'regulations', 'Sex', 'Rights', 'Discrimination', 'Law', 'Harassment', 'Policy']

for i in range(0, len(match_patterns)):
    match_patterns.append(match_patterns[i].lower())

def ismatch_title_link_map(s):
    for i in match_patterns:
        if re.search(i, s):
            return True
    return False

def add_data(text_data, s_no):
    text_list = text_data.split("\n")
    data = ''
    for text in text_list:
        if len(text) > 1 and not text.isspace():
            data = data + '\n' + text.strip()
    bytedata = data.encode('utf-8')
    binary_file = open("output/{}.txt".format(s_no), "ab")
    binary_file.write(bytedata)
    binary_file.close()
    return bytedata

def get_data_from_url(url, s_no):
    child_link = []
    try:
        html_page = requests.get(url).text
        soup = BeautifulSoup(html_page, 'html.parser')
        text_data = soup.text.strip()
        bytedata = add_data(text_data, s_no)
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
    

def find_links(url_with_depth_tuple, G, s_no):
    (url, current_depth) = url_with_depth_tuple
    print("Current Depth - ", current_depth)
    if current_depth < max_depth:
        bytedata, child_link = get_data_from_url(url, s_no)
        # G.add_node(url)
        # G.nodes[url]['data'] = bytedata
        # G.nodes[url]['child_link'] = child_link
        # print("Added Parent Node")
        for link in child_link:
            child_bytedata, child_children_link = get_data_from_url(link, s_no)
            G.add_node(link)
            G.nodes[link]['data'] = child_bytedata
            G.nodes[link]['child_link'] = child_children_link
            G.add_edge(url, link)
            # print("added edge")
            if link not in crawled_urls:
                next_urls.put((link, current_depth + 1))
                crawled_urls.append(link)
        # print("Added Child Node")

def crawl_data(next_urls, G, s_no):
    while not next_urls.empty():
        find_links(next_urls.get(), G, s_no)

# class crawler_thread(threading.Thread):
# 	def __init__(self,queue,graph):
# 		threading.Thread.__init__(self)
# 		self.to_be_crawled=queue
# 		self.graph=graph
# 	def run(self):
# 		while self.to_be_crawled.empty() is False:
# 			find_links(self.to_be_crawled.get(),self.graph)

def main():
    s_no = 1
    starting_link = "https://www2.ed.gov/about/offices/list/ocr/docs/tix_dis.html"
    to_be_crawled = []
    crawled_urls = []
    next_urls = queue.Queue()
    start_time = time.time()
    next_urls.put((starting_link, 0))
    crawled_urls.append(starting_link)
    G = nx.Graph()

    bytedata, child_link = get_data_from_url(starting_link, s_no)
    G.add_node(starting_link)
    G.nodes[starting_link]['data'] = bytedata
    G.nodes[starting_link]['child_link'] = child_link

    # thread_list=[]

    # for i in range(max_threads):
    #     t=crawler_thread(next_urls, G)
    #     t.daemon=True
    #     t.start()
    #     thread_list.append(t)

    # for t in thread_list:
    #     t.join()

    crawl_data(next_urls, G, s_no)
    nx.draw(G, with_labels = True)
    plt.savefig("output/1.png")
    write_dot(G, 'output/1.dot')
    end_time = time.time()
    print("Total Time - {}".format(end_time - start_time))

main()