pf=open('positive_frequency.txt','w')
pf.truncate()

filename='positive.txt'
total=174
frequency=[0,0,0,0] ##A:0,T:1,C:2,G:3
with open(filename,'r') as p:
	for line in p:
		A_total=0
		T_total=0
		C_total=0
		G_total=0
		line=line.strip('\n')
		for i in line:
			if i=='A':
				A_total=A_total+1
			elif i=='T':
				T_total=T_total+1
			elif i=='C':
				C_total=C_total+1
			elif i=='G':
				G_total=G_total+1
##		print(A_total)
		frequency[0]=float(A_total/total)
		frequency[1]=float(T_total/total)
		frequency[2]=float(C_total/total)
		frequency[3]=float(G_total/total)
		for j in frequency:
		##	print (j)
			pf.write(str(j))
			pf.write('	')
		pf.write('\n')

pf.close()

nf=open('negative_frequency.txt','w')
nf.truncate()
filename='negative.txt'

with open(filename,'r') as n:
	for line in n:
		A_total=0
		T_total=0
		C_total=0
		G_total=0
		line=line.strip('\n')
		for i in line:
			if i=='A':
				A_total=A_total+1
			elif i=='T':
				T_total=T_total+1
			elif i=='C':
				C_total=C_total+1
			elif i=='G':
				G_total=G_total+1
		frequency[0]=float(A_total/total)
		frequency[1]=float(T_total/total)
		frequency[2]=float(C_total/total)
		frequency[3]=float(G_total/total)
		for j in frequency:
			nf.write(str(j))
			nf.write('	')
		nf.write('\n')
			
nf.close()
