import pandas as pd 
import csv

train_df = pd.read_table('../LIAR_PLUS_Dataset/liar_train.tsv')
dev_df = pd.read_table('../LIAR_PLUS_Dataset/liar_val.tsv')
test_df = pd.read_table('../LIAR_PLUS_Dataset/liar_test.tsv')

train_lst_pre = []
test_lst_pre = []
dev_lst_pre = []

train_lst_hyp = []
test_lst_hyp = []
dev_lst_hyp = []

for i, row in train_df.iterrows():
	if row['label'] == 1:
		train_lst_pre.append([(row['statement']).encode('utf-8'), 0, row['id']])
		train_lst_hyp.append([(row['justification']).encode('utf-8'), 0, row['id']])
	elif row['label'] == 0:
		train_lst_pre.append([(row['statement']).encode('utf-8'), 1, row['id']])
		train_lst_hyp.append([(row['justification']).encode('utf-8'), 1, row['id']])

for i, row in dev_df.iterrows():
	if row['label'] == 1:
		dev_lst_pre.append([(row['statement']).encode('utf-8'), 0, row['id']])
		dev_lst_hyp.append([(row['justification']).encode('utf-8'), 0, row['id']])
	elif row['label'] == 0:
		dev_lst_pre.append([(row['statement']).encode('utf-8'), 1, row['id']])
		dev_lst_hyp.append([(row['justification']).encode('utf-8'), 1, row['id']])

for i, row in test_df.iterrows():
	if row['label'] == 1:
		test_lst_pre.append([(row['statement']).encode('utf-8'), 0, row['id']])
		test_lst_hyp.append([(row['justification']).encode('utf-8'), 0, row['id']])
	elif row['label'] == 0:
		test_lst_pre.append([(row['statement']).encode('utf-8'), 1, row['id']])
		test_lst_hyp.append([(row['justification']).encode('utf-8'), 1, row['id']])

with open("data/liar_em_train_pre.tsv", 'w', newline = '') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in train_lst_pre:
		tsv_writer.writerow(obj)

with open("data/liar_em_test_pre.tsv", 'w', newline = '') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in test_lst_pre:
		tsv_writer.writerow(obj)

with open("data/liar_em_val_pre.tsv", 'w', newline = '') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in dev_lst_pre:
		tsv_writer.writerow(obj)

with open("data/liar_em_train_hyp.tsv", 'w', newline = '') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in train_lst_hyp:
		tsv_writer.writerow(obj)

with open("data/liar_em_test_hyp.tsv", 'w', newline = '') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in test_lst_hyp:
		tsv_writer.writerow(obj)

with open("data/liar_em_val_hyp.tsv", 'w', newline = '') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in dev_lst_hyp:
		tsv_writer.writerow(obj)