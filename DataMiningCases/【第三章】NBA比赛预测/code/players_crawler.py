
import socket
import urllib
import time
import os
from bs4 import BeautifulSoup

socket.setdefaulttimeout(20)



prefix = 'http://www.basketball-reference.com'



def get_data(url, proxy = None):
	proxies = {'http':proxy}
	if url.startswith('/'):
		url = prefix + url
	print url
	while True:
		try:
			if proxy:
				return urllib.urlopen(url, proxies=proxies).read()
			else:
				return urllib.urlopen(url).read()
		except:
			print 'error when fetching web data!'
			time.sleep(5)

def write_to_file(filepath, content):
	folderpath = os.path.dirname(filepath)
	if not os.path.exists(folderpath):
		os.makedirs(folderpath)
	f = open(filepath, 'w')
	f.write(content)
	f.close()



for i in range(26):
	url = prefix + '/players/%c/' % (97+i)
	s = get_data(url)
	players_soup = BeautifulSoup(s)

	output = ''

	table = players_soup.table
	if table==None: continue
	trs = table.findAll('tr')

	tr = trs[0]
	p_ths = map(lambda x: x.text, tr.findAll('th'))
	p_ths.insert(0, 'Player ID')
	output += ','.join(p_ths) + '\n'
	del trs[0]

	for tr in trs:
		tds = tr.findAll('td')
		p_tds = map(lambda x: x.text, tds)
		p_tds.insert(0, os.path.basename(tds[0].a.get('href'))[:-5])
		p_tds[1] = tds[0].a.text
		p_tds[7] = '"'+p_tds[7]+'"'
		p_tds[8] = '"'+p_tds[8]+'"'
		output += ','.join(p_tds) + '\n'

	filepath = 'players/%c/players.csv' % (97+i)
	write_to_file(filepath, output)


