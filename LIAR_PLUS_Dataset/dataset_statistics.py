import pandas as pd
from collections import Counter
train_data = pd.read_table('liar_train.tsv')

val_data = pd.read_table('liar_val.tsv')

test_data = pd.read_table('liar_test.tsv')

def give_statistics(ip_df):
	label_distribution = Counter(ip_df['label'])
	statement_lt = 0
	justification_lt = 0
	total = 0
	for i, row in ip_df.iterrows():
		st_lt = len(row['statement'].split(' '))
		jt_lt = len(row['justification'].split(' '))
		statement_lt+=(st_lt)
		justification_lt+=(jt_lt)
		total+=1
	avg_st_lt = statement_lt / total
	avg_jt_lt = justification_lt / total
	return label_distribution, avg_st_lt, avg_jt_lt

train_ld, train_st, train_jt = give_statistics(train_data)
val_ld, val_st, val_jt = give_statistics(val_data)
test_ld, test_st, test_jt = give_statistics(test_data)

with open("Dataset_statistics.txt", 'w') as outfile:
	print("######TRAIN######", file = outfile)
	print("Label Count", train_ld, file = outfile)
	print("Statement Avg Length", train_st, file = outfile)
	print("Justification Avg Length", train_jt, file = outfile)
	print("######TEST######", file = outfile)
	print("Label Count", test_ld, file = outfile)
	print("Statement Avg Length", test_st, file = outfile)
	print("Justification Avg Length", test_jt, file = outfile)
	print("######VAL######", file = outfile)
	print("Label Count", val_ld, file = outfile)
	print("Statement Avg Length", val_st, file = outfile)
	print("Justification Avg Length", val_jt, file = outfile)