import sys, os, shutil

from argparse import ArgumentParser
import re

import urllib
import urllib2
from bs4 import BeautifulSoup
from urlparse import urljoin
from posixpath import basename

import threading

max_thread = 20
sema = threading.Semaphore(max_thread)

def html_downloader(url, next_htmls, folderName):
    try:
        response = urllib2.urlopen(url)

        # check if the url is a valid html website
        if "text/html" in response.headers["content-type"]:
            htmlStr = response.read()
            next_htmls.append((url, htmlStr))

        elif '.mid' in url:
            file = open(folderName+'\\'+basename(url), 'wb')
            file.write(response.read())
            file.close()
            sys.stdout.write('.')
    except:
        pass

    sema.release()
    exit()

def titleFinder(url, visited_urls, htmlStr, folderName, urlRegex=None, depth=1):
    print '\n'+url
    soup = BeautifulSoup(htmlStr, "html.parser")

    # download the htmls
    thread_list = []
    next_htmls = []
    for link in soup.findAll('a'):
        if link.has_attr('href'):
            new_path = urljoin(url, link['href'])

            if (urlRegex!=None):
                regexPass= False
                for regEx in urlRegex:
                    if (re.search(regEx, new_path)!=None):
                        regexPass = True
                        break
                if not regexPass:
                    continue

            if (new_path not in visited_urls) and (depth!=0 or ('.mid' in new_path)):
                visited_urls.append(new_path)

                sema.acquire(True)
                th = threading.Thread(target=html_downloader, args=(new_path, next_htmls, folderName))

                thread_list.append(th)
                th.start()

    # wait for the threads to finish
    for th in thread_list:
        th.join()

    if depth==0:
        return

    # keep on crawling
    for nextURL,nextResp in next_htmls:
        titleFinder(nextURL, visited_urls, nextResp, folderName, urlRegex, depth=depth-1)

def parseCLI():
    desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)

    parser = ArgumentParser(description=desc)

    parser.add_argument('-u', '--url', type = str, dest = 'url', required = True,
                        help = 'URL of the website to crawl in')
    parser.add_argument('-f', '--folderName', type = str, dest = 'folderName', required = True,
                        help = 'MIDI output folder')
    parser.add_argument('-d', '--depth', type = int, dest = 'depth', required = True,
                        help = 'Crawl depth')
    parser.add_argument('-r', '--urlRegex', type = str, dest = 'urlRegex',
                        help = 'RegEx urls need to follow')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseCLI()

    # make the directory for the created files
    try:
        os.mkdir(args.folderName)
    except:
        pass

    if args.urlRegex!=None:
        args.urlRegex = args.urlRegex.split(',')

    # start crawling
    visited_urls = []
    response = urllib2.urlopen(args.url)
    html = response.read()
    titleFinder(args.url, visited_urls, html, args.folderName, args.urlRegex, depth=args.depth)
    
    print "done"