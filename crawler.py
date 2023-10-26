import getopt
import os
import socket
import subprocess
import sys
import threading
import queue
import urllib
import bs4
import networkx as nx
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
global parser_flag, crawled_urls, next_url, max_depth
from urllib.parse import urlparse

max_threads=50
crawled_urls=[]
next_url=queue.Queue()
root_url="https://www2.ed.gov/about/offices/list/ocr/docs/tix_dis.html"
parser_flag = 'beautifulsoup'
max_depth=10

def check_link(url):
    print("URL - ", urlparse(url))
    domain='.'.join(urlparse(url).netloc.split('.')[-2:])
    print("Domain - ", domain)
    if domain == 'ed.gov':
        return True
    return False

def get_links_from_page(url):
    urllist = []
    try:
        res=urllib2.urlopen(url)
        htmlpage=res.read()
    except:
        return urllist

    try:
        page=BeautifulSoup.BeautifulSoup(htmlpage)
    except:
        return urllist

    refs=page.findAll("a")
    for a in refs:
        try:
            link = a['href']
            if link[:4] == 'http':
                urllist.append(link)
        except:
            pass
    return urllist

def find_links(url_tuple, graph):
    url = url_tuple[0]
    depth = url_tuple[1]
    if depth < max_depth:
        links = get_links_from_page(url)
        for link in links:
            graph.add_node(link)
            graph.add_edge(url, link)
            if link not in crawled_urls:
                next_url.put((link, depth + 1))
                crawled_urls.append(link)
    return

class crawler_thread(threading.Thread):
	def __init__(self,queue,graph):
		threading.Thread.__init__(self)
		self.to_be_crawled=queue
		self.graph=graph
	def run(self):
		while self.to_be_crawled.empty() is False:
			find_links(self.to_be_crawled.get(),self.graph)



next_url.put((root_url,0))
crawled_urls.append(root_url)
ip_list=[]
g=nx.Graph()
g.add_node(root_url)
thread_list=[]

for i in range(max_threads):
    t=crawler_thread(next_url,g)
    t.daemon=True
    t.start()
    thread_list.append(t)

for t in thread_list:
    t.join()

for url in crawled_urls:
    ip_list.append(socket.gethostbyname(urlparse(url).netloc))
    ip_list=list(set(ip_list))

print("Unique Host: %s " % len(ip_list))

fh=open(os.getcwd()+'/targets.list','w')
for ip in ip_list:
    fh.write(str(ip)+'\n')

nodesize=[g.degree(n)*10 for n in g]
pos=nx.spring_layout(g,iterations=20)
nx.draw(g,with_labels=False)
nx.draw_networkx_nodes(g,pos,node_size=nodesize,node_color='r')
nx.draw_networkx_edges(g,pos)
plt.show()
plt.savefig("crawl.png")
nx.write_dot(g,"crawl.dot")