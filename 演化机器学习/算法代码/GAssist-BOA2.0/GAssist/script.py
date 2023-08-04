import os
for i in range (0,2):
	print os.system('java -jar -Xmx1024m MDL-GAssist-Match-Iter.jar command1.txt 1-result.txt')
for i in range (0,2):
	print os.system('java -jar -Xmx1024m MDL-GAssist-Match-Iter.jar command2.txt 2-result.txt')
for i in range (2,4):
	print os.system('java -jar -Xmx1024m MDL-GAssist-Match-Iter.jar command2.txt 1-result.txt')
for i in range (2,4):
	print os.system('java -jar -Xmx1024m MDL-GAssist-Match-Iter.jar command1.txt 2-result.txt')