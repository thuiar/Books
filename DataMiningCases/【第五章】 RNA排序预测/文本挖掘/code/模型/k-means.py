from sklearn import cluster
import numpy as np

plen=0
save_positive_sequence=[]
with open ('positive_number.txt','r') as pf:
	for line in pf:
		line=line.strip('\n')
		save_positive_sequence.append(line)
		plen+=1
print(plen)

save_negative_sequence=[]
with open('negative_number.txt','r')as nf:
	for line in nf:
		line=line.strip('\n')
		save_negative_sequence.append(line)


dataset=np.loadtxt('total_frequency.txt')
total_datanumber=np.shape(dataset)[0]
k_means=cluster.KMeans(3)
k_means.fit(dataset)

save_cluster=k_means.labels_

p_dataset1=open('positive_dataset1.txt','w')
p_dataset1.truncate()
p_dataset2=open('positive_dataset2.txt','w')
p_dataset2.truncate()
p_dataset3=open('positive_dataset3.txt','w')
p_dataset3.truncate()
n_dataset1=open('negative_dataset1.txt','w')
n_dataset1.truncate()
n_dataset2=open('negative_dataset2.txt','w')
n_dataset2.truncate()
n_dataset3=open('negative_dataset3.txt','w')
n_dataset3.truncate()

for i in range(plen):
	if (save_cluster[i]==0):
		p_dataset1.write(save_positive_sequence[i])
		p_dataset1.write('\n')
	elif (save_cluster[i]==1):
		p_dataset2.write(save_positive_sequence[i])
		p_dataset2.write('\n')
	else:
		p_dataset3.write(save_positive_sequence[i])
		p_dataset3.write('\n')

for i in range(plen,total_datanumber):
	if (save_cluster[i]==0):
		n_dataset1.write(save_negative_sequence[i-plen])
		n_dataset1.write('\n')
	elif (save_cluster[i]==1):
		n_dataset2.write(save_negative_sequence[i-plen])
		n_dataset2.write('\n')
	else:
		n_dataset3.write(save_negative_sequence[i-plen])
		n_dataset3.write('\n')

p_dataset1.close()
p_dataset2.close()
p_dataset3.close()
n_dataset1.close()
n_dataset2.close()
n_dataset3.close()

