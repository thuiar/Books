myfile='initial_positive.txt'
save_seq=[]
with open(myfile,'r') as f:
	for line in f:
		line=line.strip('\n')
		if line not in save_seq:
			save_seq.append(line)

pf=open('save_for_fasta.txt','w')
pf.truncate()
for i in save_seq:
	pf.write('>sequence')
	pf.write(str(save_seq.index(i)))
	pf.write('\n')
	pf.write(i)
	pf.write('\n')
pf.close()
