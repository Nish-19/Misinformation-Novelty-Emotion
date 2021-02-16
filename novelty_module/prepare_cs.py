import pandas as pd

def prepare_dataset(ip_df):
	lst = []
	premise = 'chloroquine hydroxychloroquine are cure for the novel coronavirus'
	# 1 - Against (disagree)
	# 2 - For (agree)
	a_ctr=0
	f_ctr=0
	for i, row in ip_df.iterrows():
		if row['stance'] == 1:
			a_ctr+=1
			lst.append([premise, row['text'], 0])
		elif row['stance'] == 2:
			f_ctr+=1
			lst.append([premise, row['text'], 1])

	with open(file.split('.')[0]+"_quora_convert.txt", 'w') as outfile:
		for obj in lst:
			outfile.write(str(obj[0]) + '\t' + str(obj[1]) + '\t' + str(obj[2]) + '\n')

file = '../Covid_Stance_Dataset/cstance_train.csv'
ip_df = pd.read_csv(file)
prepare_dataset(ip_df)

file = '../Covid_Stance_Dataset/cstance_test_new.csv'
ip_df = pd.read_csv(file)
prepare_dataset(ip_df)

data = data2 = "" 
  
# Reading data from file1 
with open('data/quora/train.txt', encoding='utf-8') as fp: 
    data = fp.read()
  
# Reading data from file2 
with open('cstance_test_new_quora_convert.txt', encoding='utf-8') as fp: 
    data2 = fp.read()
  
# Merging 2 files 
# To add the data of file2 
# from next line 
data += "\n"
data += data2 
  
with open ('combined_train_cs.txt', 'w', encoding='utf-8') as fp: 
    fp.write(data) 