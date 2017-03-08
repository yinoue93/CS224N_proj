import sys, os, shutil

from argparse import ArgumentParser
import re

import urllib
import urllib2
from bs4 import BeautifulSoup
from urlparse import urljoin
from posixpath import basename

import threading

max_thread = 10
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

counter = 0
def thesession_downloader(pageNum, folderName):
	try:
		base_url = 'https://thesession.org'
		url = base_url+'/tunes/'+str(pageNum)

		response = urllib2.urlopen(url)

		soup = BeautifulSoup(response.read(), "html.parser")

		songName = soup.find('h1').getText()
		count = 0
		for link in soup.findAll('a'):
			if link.has_attr('href'):
				if re.match('/tunes/%s/abc/[0-9]+' % pageNum, link['href']):
					if os.path.exists('%s\\%s_%d.abc' %(folderName,songName,count)):
						continue

					abcUrl = base_url+link['href']

					file = open('%s\\%s_%d.abc' %(folderName,songName,count), 'wb')
					abcResponse = urllib2.urlopen(abcUrl)
					file.write(abcResponse.read())
					file.close()
					sys.stdout.write('.')

					count += 1

	except:
		sys.stdout.write('X')
		pass

	global counter
	counter += 1
	print counter
	sema.release()
	exit()

def scrapeTheSession(folderName):
	thread_list = []
	for i in xrange(16000,17000):
		sema.acquire(True)
		th = threading.Thread(target=thesession_downloader, args=(i, folderName))

		thread_list.append(th)
		th.start()

	# wait for the threads to finish
	for th in thread_list:
		th.join()

def montreal_downloader(url, outputname):
	try:
		response = urllib2.urlopen(url)
		soup = BeautifulSoup(response.read(), "html.parser")

		abcTxt = soup.find('div', {'class':'abc'})

		if abcTxt!=None:
			with open(outputname, 'wb') as abcf:
				abcf.write(re.sub(r'[^\x00-\x7f]',r'',abcTxt.getText()))
		
		sys.stdout.write('.')

	except:
		sys.stdout.write('X')
		pass

	sema.release()

def scrapeTheMontreal(folderName):
	# first find the urls of all songs
	url = 'http://music.gordfisch.net/montrealsession/complete.php'
	response = urllib2.urlopen(url)
	soup = BeautifulSoup(response.read(), "html.parser")

	song_urls = []
	song_names = []
	for link in soup.findAll('a', href=True):
		song_name = re.sub(r'[^\x00-\x7f]',r'',link.getText())
		if len(song_name)<3:
			continue

		song_name = song_name.replace('.','').replace(',','')+'.abc'
		count = 0
		while song_name.replace('.abc','_%d.abc'%count) in song_names:
			count += 1
		song_names.append(song_name.replace('.abc','_%d.abc'%count))
		song_urls.append(urljoin(url, link['href']))

	thread_list = []
	for song_url,song_name in zip(song_urls,song_names):
		sema.acquire(True)
		th = threading.Thread(target=montreal_downloader, 
							  args=(song_url, os.path.join(folderName,song_name)))

		thread_list.append(th)
		th.start()

	# wait for the threads to finish
	for th in thread_list:
		th.join()

def scrapeLocally(url, folderName):
	"""
	Scrapes .abc files stored locally under @url.
	@url should be a file with multiple .abc files listed.
	"""
	for fname in os.listdir(url):
		with open(os.path.join(url,fname),'r') as abcfile:
			count = 0
			during_file = False

			for line in abcfile:
				line = line.strip()
				if not during_file and line.replace(' ','')[:2]=='X:':
					during_file = True
					newFile = '%s_%d.abc' % (fname,count)
					count += 1
					f = open(os.path.join(folderName,newFile), 'w')

				if during_file:
					if len(line)==0:
						during_file = False
						f.close()
					else:
						f.write(line+'\n')


def parseCLI():
	desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)

	parser = ArgumentParser(description=desc)

	parser.add_argument('-u', '--url', type = str, dest = 'url', required = True,
						help = 'URL of the website to crawl in')
	parser.add_argument('-f', '--folderName', type = str, dest = 'folderName', required = True,
						help = 'MIDI output folder')
	parser.add_argument('-d', '--depth', type = int, dest = 'depth', default=2,
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
	if 'thesession.org' in args.url:
		scrapeTheSession(args.folderName)
	elif 'montreal' in args.url:
		scrapeTheMontreal(args.folderName)
	elif ('http' not in args.url) and ('www' not in args.url):
		scrapeLocally(args.url, args.folderName)
	else:
		visited_urls = []
		response = urllib2.urlopen(args.url)
		html = response.read()
		titleFinder(args.url, visited_urls, html, args.folderName, args.urlRegex, depth=args.depth)
	
	print "done"