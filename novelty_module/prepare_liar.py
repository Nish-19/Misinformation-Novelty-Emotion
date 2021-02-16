import pandas as pd 
# column_names = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
#                                             "barely-true", "false", "half-true", "mostly-true", "pants-fire", "venue", "justification"]
train_data = pd.read_table('new_dataset/liar_train.tsv')

val_data = pd.read_table('new_dataset/liar_val.tsv')

test_data = pd.read_table('new_dataset/liar_test.tsv')
# 0 - Novel
# 1 - Duplicate
def create_dataset(ip_df, name):
	lst = []
	err_lst = []
	for i, row in ip_df.iterrows():
		print(i)
		# For False
		if row['label'] == 0:
			label = 0
		# For True
		elif row['label'] == 1:
			label = 1
		else:
			print("Label not matched")
		lst.append([row['statement'].encode('utf-8'), row['justification'].encode('utf-8'), label])
	with open(name, 'w', newline = '') as outfile:
		for obj in lst:
			outfile.write(str(obj[0]) + '\t' + str(obj[1]) + '\t' + str(obj[2]) + '\n')
	print("Error List values are", err_lst)
	return err_lst

train_err = create_dataset(train_data, 'liar_train.txt')
test_err = create_dataset(test_data, 'liar_test.txt')
val_err = create_dataset(val_data, 'liar_dev.txt')
with open("Error_Entries.txt", 'w') as outfile:
	print("Train Errors", train_err, file = outfile)
	print("Test Errors", test_err, file = outfile)
	print("Dev Errors", val_err, file = outfile)

data = data2 = "" 
  
# Reading data from file1 
with open('data/quora/train.txt', encoding='utf-8') as fp: 
    data = fp.read()
  
# Reading data from file2 
with open('liar_dev.txt', encoding='utf-8') as fp: 
    data2 = fp.read()
  
# Merging 2 files 
# To add the data of file2 
# from next line 
data += "\n"
data += data2 
  
with open ('combined_train_liar.txt', 'w', encoding='utf-8') as fp: 
    fp.write(data) 