#-*- coding:utf-8 -*-
import os
import csv

header = 'GAME,MP,FG,FGA,FG%,3P,3PA,3P%,FT,FTA,FT%,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS,HOME,WIN'.split(',')

def maketeaminfo(gamedir, teaminfodir):
	teamwriter = {}
	for gamefilename in os.listdir(gamedir):
		gamefile = file(os.path.join(gamedir, gamefilename), 'rb')
		gamereader = csv.reader(gamefile)
		team = ['', '']
		data = [[], []]
		for line in gamereader:
			if len(line) == 1:
				if team[0] == '':
					team[0] = line[0]
				else:
					team[1] = line[0]
				if line[0] not in teamwriter:
					teamwriter[line[0]] = csv.writer(open(os.path.join(teaminfodir, line[0] + '.csv'), 'wb'))
					teamwriter[line[0]].writerow(header)
			elif line[0] == 'Team Totals':
				if line[-1] == '':
					line.pop()
				line[0] = gamefilename[0:-4]
				if len(data[0]) is 0:
					line.append('0')
					data[0] = line
				else: 
					line.append('1')
					data[1] = line
		if int(data[0][-2]) > int(data[1][-2]):
			data[0].append('1')
			data[1].append('0')
		else:
			data[0].append('0')
			data[1].append('1')
		teamwriter[team[0]].writerow(data[0])
		teamwriter[team[1]].writerow(data[1])
		gamefile.close()

maketeaminfo(u'F:/傅汪/project/球赛预测/data/games/2012/playoff', 'playoff')