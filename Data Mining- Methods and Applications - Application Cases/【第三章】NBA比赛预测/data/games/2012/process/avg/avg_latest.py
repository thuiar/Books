#-*- coding:utf-8 -*-
import csv
import os

header = 'TEAM,FG,FGA,FGP,3P,3PA,3PP,FT,FTA,FTP,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS,HOME,WIN'.split(
    ',')


def avg_latest(teaminfodir, avgfile, latest):
    gamedata = {}
    for team in os.listdir(teaminfodir):
        teaminfo = open(os.path.join(teaminfodir, team), 'rb')
        team = team[0:-4]
        teaminforeader = csv.reader(teaminfo)
        teaminfoheader = teaminforeader.next()
        sumdata = {
            'FG': 0, 'FGA': 0, '3P': 0, '3PA': 0, 'FT': 0, 'FTA': 0, 'ORB': 0, 'DRB': 0,
            'TRB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0, 'PF': 0, 'PTS': 0, 'HOME': 0, 'WIN': 0}
        queue = []
        for line in teaminforeader:
            if latest < 0 and len(queue) > 0:
                count = len(queue)
            else:
                count = latest
            if len(queue) == count:
                data = (
                    team, sumdata['FG'] / float(count), sumdata['FGA'] / float(
                        count), float(sumdata['FG']) / sumdata['FGA'],
                    sumdata['3P'] / float(count), sumdata['3PA'] / float(
                        count), float(sumdata['3P']) / sumdata['3PA'],
                    sumdata['FT'] / float(count), sumdata['FTA'] / float(
                        count), float(sumdata['FT']) / sumdata['FTA'],
                    sumdata['ORB'] / float(count), sumdata['DRB'] / float(
                        count), sumdata[
                            'TRB'] / float(count), sumdata['AST'] / float(count),
                    sumdata['STL'] / float(count), sumdata['BLK'] / float(
                        count), sumdata[
                            'TOV'] / float(count), sumdata['PF'] / float(count),
                    sumdata['PTS'] / float(count), sumdata['HOME'] / float(count), sumdata['WIN'] / float(count))
                if line[0] not in gamedata:
                    gamedata[line[0]] = [(), (), int(line[-1] == line[-2])]
                gamedata[line[0]][int(line[-2])] = data
            queue.append(line)
            for key, value in zip(teaminfoheader, line):
                if key in sumdata.keys():
                    sumdata[key] += int(value)
            if len(queue) == (latest + 1):
                overtime = queue.pop(0)
                for key, value in zip(teaminfoheader, overtime):
                    if key in sumdata.keys():
                        sumdata[key] -= int(value)
    writer = csv.writer(open(avgfile, 'wb'))
    wheader = ['GAME']
    for attr in header:
        wheader.append('R_' + attr)
    for attr in header:
        wheader.append('H_' + attr)
    wheader.append('RESULT')
    writer.writerow(wheader)
    for key in gamedata.keys():
        data = gamedata[key]
        if len(data[0]) == 0 or len(data[1]) == 0:
            continue
        writer.writerow((key,) + data[0] + data[1] + (data[2],))


def merge(csv1, pre1, csv2, pre2, outcsv):
    instance = {}
    avg1 = csv.reader(open(csv1, 'rb'))
    avg2 = csv.reader(open(csv2, 'rb'))
    header = avg1.next()
    header = header[1:-1]
    for line in avg1:
        instance[line[0]] = line
    writer = csv.writer(open(outcsv, 'wb'))
    wheader = ['GAME']
    wheader.append(header[1])
    for attr in header[1:len(header) / 2]:
        wheader.append(pre1 + attr)
    for attr in header[1:len(header) / 2]:
        wheader.append(pre2 + attr)
    wheader.append(header[len(header) / 2])
    for attr in header[len(header) / 2 + 1:]:
        wheader.append(pre1 + attr)
    for attr in header[len(header) / 2 + 1:]:
        wheader.append(pre2 + attr)
    wheader.append('RESULT')
    writer.writerow(wheader)
    sinlen = len(header) / 2
    for line in avg2:
        if line[0] in instance:
            newins = instance[line[0]][0:1 + sinlen] + line[
                2:1 + sinlen] + instance[line[0]][1 + sinlen:-1] + line[2 + sinlen:]
            writer.writerow(newins)

#avg_latest(u'F:/傅汪/project/球赛预测/data/games/2012/process/teaminfo/regular_season', 'regular_season/avg1.csv', 1)
#avg_latest(u'F:/傅汪/project/球赛预测/data/games/2012/process/teaminfo/regular_season', 'regular_season/avg5.csv', 5)
merge('regular_season/avgall.csv', 'avg_', 'regular_season/avg1.csv',
      'avg1_', 'regular_season/merge_avg_avg1.csv')
