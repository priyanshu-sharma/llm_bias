import queue
import re
import time
from urllib.parse import urljoin

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests
from bs4 import BeautifulSoup
from networkx.drawing.nx_pydot import write_dot


class CrawlGraph:
    def __init__(self, s_no, starting_link, level):
        self.s_no = s_no
        self.starting_link = starting_link
        self.level = level
        self.max_depth = 1
        self.initial_patterns = ['U.S. Department', 'Title IX', 'Commission', 'regulations', 'Sex', 'Rights', 'Discrimination', 'Law', 'Harassment', 'Policy']
        self.match_patterns = self.initialize_patterns()
        self.to_be_crawled = []
        self.crawled_urls = []
        self.next_urls = queue.Queue()
        self.crawl()

    def initialize_patterns(self):
        final_patterns = self.initial_patterns
        for i in range(0, len(self.initial_patterns)):
            final_patterns.append(self.initial_patterns[i].lower())
        return final_patterns

    def ismatch_title_link_map(self, s):
        for i in self.match_patterns:
            if re.search(i, s):
                return True
        return False

    def add_data(self, text_data):
        text_list = text_data.split("\n")
        data = ''
        for text in text_list:
            if len(text) > 1 and not text.isspace():
                data = data + '\n' + text.strip()
        bytedata = data.encode('utf-8')
        binary_file = open("output/{}/{}.txt".format(self.level, self.s_no), "ab")
        binary_file.write(bytedata)
        binary_file.close()
        return bytedata

    def get_data_from_url(self, url):
        child_link = []
        # try:
        html_page = requests.get(url).text
        soup = BeautifulSoup(html_page, 'html.parser')
        text_data = soup.text.strip()
        bytedata = self.add_data(text_data)
        sublinks = soup.find_all("a")
        for sublink in sublinks:
            if self.ismatch_title_link_map(sublink.text.strip()):
                if sublink.get("href") and sublink["href"][0] == '/':
                    path = urljoin(url, sublink["href"])
                    child_link.append(path)
                elif sublink.get("href") and sublink["href"][:4] == 'http':
                    path = sublink["href"]
                    child_link.append(path)
        return bytedata, child_link
        # except Exception as e:
        #     print("Exception - ", e) 

    def find_links(self, url_with_depth_tuple, G):
        (url, current_depth) = url_with_depth_tuple
        print("Current Depth - ", current_depth)
        if current_depth < self.max_depth:
            bytedata, child_link = self.get_data_from_url(url)
            for link in child_link:
                child_bytedata, child_children_link = self.get_data_from_url(link)
                G.add_node(link)
                G.nodes[link]['data'] = child_bytedata
                G.nodes[link]['child_link'] = child_children_link
                G.add_edge(url, link)
                if link not in self.crawled_urls:
                    self.next_urls.put((link, current_depth + 1))
                    self.crawled_urls.append(link)

    def crawl_data(self, G):
        while not self.next_urls.empty():
            self.find_links(self.next_urls.get(), G)

    def crawl(self):
        self.next_urls.put((self.starting_link, 0))
        self.crawled_urls.append(self.starting_link)
        G = nx.Graph()
        bytedata, child_link = self.get_data_from_url(self.starting_link)
        G.add_node(self.starting_link)
        G.nodes[self.starting_link]['data'] = bytedata
        G.nodes[self.starting_link]['child_link'] = child_link
        self.crawl_data(G)
        nx.draw(G, with_labels = True)
        plt.savefig("output/{}/{}.png".format(self.level, self.s_no))
        write_dot(G, 'output/{}/{}.dot'.format(self.level, self.s_no))


def starter():
    df = pd.read_csv('new_data.csv')
    print(df.head(10))
    for _, row in df.iterrows():
        print(row['s_no'], row['url'], row['level'])
        print('----------------------------------')
        start_time = time.time()
        try:
            crawlgraph = CrawlGraph(row['s_no'], row['url'], row['level'])
        except Exception as e:
            print("Exception - ", e)
        end_time = time.time()
        print("Total Time - {}".format(end_time - start_time))
        print('----------------------------------')
        time.sleep(2)

starter()

# 101, 81, 74, 68
# 68,https://malegislature.gov/Laws/GeneralLaws/PartI/TitleIX/Chapter59/Section5A,mass,-1,-1,-1,-1