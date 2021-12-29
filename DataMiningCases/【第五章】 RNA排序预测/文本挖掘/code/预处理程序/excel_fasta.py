import xlrd

############get ires_name,gfp_expression,oligo_sequence############
file_name='3.xlsx'      ######also 3,4,5,6,7####

data=xlrd.open_workbook(file_name)
table=data.sheets()[0]
ncols=table.ncols
nrows=table.nrows

def find_col_index(index_name):
	return_index=0
	for i in range(ncols):
		if table.row(0)[i].value==index_name:
			return_index=i
	return return_index

ires_name_index=find_col_index('Oligo_Index')
score_index=find_col_index('eGFP_expression (a.u)')
sequence_index=find_col_index('Oligo_sequence')

ires_dictionary={}
for i in range(nrows):
	if (isinstance(table.row(i)[score_index].value,float)):
		if table.row(i)[score_index].value>600:
			if table.row(i)[ires_name_index].value not in ires_dictionary:
				ires_dictionary[table.row(i)[ires_name_index].value]=table.row(i)[sequence_index].value
		##	else:
		##		ires_dictionary[table.row(i)[ires_name_index].value] += table.row(i)[sequence_index].value

negative_dictionary={}
for i in range(nrows):
	if (isinstance(table.row(i)[score_index].value,float)):
		if table.row(i)[score_index].value<=600:
			if table.row(i)[ires_name_index].value not in negative_dictionary:
				negative_dictionary[table.row(i)[ires_name_index].value]=table.row(i)[sequence_index].value
		##	else:
		##		negative_dictionary[table.row(i)[ires_name_index].value] += table.row(i)[sequence_index].value
	elif (table.row(i)[score_index].value=='NAN'):
		if table.row(i)[ires_name_index].value not in negative_dictionary:
			negative_dictionary[table.row(i)[ires_name_index].value]=table.row(i)[sequence_index].value
	##	else:
	##		negative_dictionary[table.row(i)[ires_name_index].value] += table.row(i)[sequence_index].value
			
##################CTAGGGCGCGCCAGTCCT################CGACTCGGACCGATGGTGAG#########################
pos=open('positive.txt','w')
pos.truncate()
for i in ires_dictionary:
##	pos.write('>sequence')
##	pos.write(str(i))
##	pos.write('\n')
	pos.write(ires_dictionary[i])
	pos.write('\n')
pos.close()

neg=open('negative.txt','w')
neg.truncate()
for i in negative_dictionary:
##	neg.write('>sequence')
##	neg.write(str(i))
##	neg.write('\n')
	neg.write(negative_dictionary[i])
	neg.write('\n')
neg.close()
