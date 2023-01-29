
import sys
import socket
import urllib
import time
import os
from bs4 import BeautifulSoup

socket.setdefaulttimeout(20)


year = int(sys.argv[1])

prefix = 'http://www.basketball-reference.com'



def get_data(url, proxy = None):
	proxies = {'http':proxy}
	if url.startswith('/'):
		url = prefix + url
	print url
	sys.stdout.flush()
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

def process_games(games_category, games_table):
	texts = games_table.findAll(text='Box Score')
	for text in texts:
		a = text.findParent()
		link = a.get('href')
		s = get_data(link)
		game_soup = BeautifulSoup(s)

		output = ''

		spans = game_soup.findAll('span', attrs={'class':'bold_text large_text'})
		teams = map(lambda x: x.a.text, spans)
		tables = map(lambda x: x.findParent('table'), game_soup.findAll(text='Basic Box Score Stats'))

		output += teams[0] + '\n'
		table = tables[0]
		trs = table.findAll('tr')
		tr = trs[1]
		p_ths = map(lambda x: x.text, tr.findAll('th'))
		p_ths[0] = 'Players'
		output += ','.join(p_ths) + '\n'
		for tr in trs:
			tds = tr.findAll('td')
			if len(tds)==0: continue
			p_tds = map(lambda x: x.text, tds)
			if tds[0].a!=None:
				p_tds[0] = os.path.basename(tds[0].a.get('href'))[:-5]
			output += ','.join(p_tds) + '\n'

		output += teams[1] + '\n'
		table = tables[1]
		trs = table.findAll('tr')
		tr = trs[1]
		p_ths = map(lambda x: x.text, tr.findAll('th'))
		p_ths[0] = 'Players'
		output += ','.join(p_ths) + '\n'
		for tr in trs:
			tds = tr.findAll('td')
			if len(tds)==0: continue
			p_tds = map(lambda x: x.text, tds)
			if tds[0].a!=None:
				p_tds[0] = os.path.basename(tds[0].a.get('href'))[:-5]
			output += ','.join(p_tds) + '\n'

		filepath = '%s/%s.csv' % (games_category, os.path.basename(link)[:-5])
		write_to_file(filepath, output)



s = get_data('http://www.basketball-reference.com/leagues/')
leagues_soup = BeautifulSoup(s)

year_str = '%.4d-%.2d' % (year, (year+1) % 100)
a = leagues_soup.find(text=year_str).findParent()
tr = a.findParent().findParent() # other information can be retrieved from <tr/>
link = a.get('href')[:-5] + '_games.html'
s = get_data(link)

games_soup = BeautifulSoup(s)
rs_table, po_table = games_soup.findAll('table')

process_games('games/%.4d/regular_season' % year, rs_table)
process_games('games/%.4d/playoff' % year, po_table)


