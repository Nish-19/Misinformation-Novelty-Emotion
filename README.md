# Misinformation_Multitask_NES
Code for the paper **"What the Fake? Probing Misinformation Standing on the Shoulder of Novelty and Emotion."**

Rina Kumari*, Nischal A\*, Tirthankar Ghoshal, Asif Ekbal

\* denotes equal contribution


To replicate the results please follow the following instructions.

## Generating Novelty Results

Place the glove-300d embeddings from the resources folder of the main directory to novelty_module/resources

### Preprocessing -
Run python preprocess_bytedance.py
Run python preprocess_fnc.py
And place the generated combined files in a folder data/quora_bd and data/quora_fnc_4ag_5dg respectively along with the test.txt and dev.txt from the data/quora folder

### Preparing the datasets for evaluation
Run all_convert_txt.py and place the fnc results in the folder data/fnc_quora
and bytedance results in data/bd_quora.py

- ByteDance Dataset
1) Training the model with the Quora-BD train data
python novelty_module/train.py novelty_module/configs/main_quora.json5
2) Generating the novelty aware embeddings using trained model (separately from train and test data)
Note - Please change the name of the output representations and predictions file every time you run with a different dataset - to change line 122 and line 126 of the src/model_new.py file
python novelty_module/evaluate.py novelty_module/models/quora_bd/benchmark/best.pt novelty_module/data/bd_quora/train.txt

- FNC Dataset
1) Training the model with the Quora-BD train data
python novelty_module/train.py novelty_module/configs/main_quora_new.json5
2) Generating the novelty aware embeddings using trained model (separately from train and test data)
python novelty_module/evaluate.py novelty_module/models/quora_fnc_4ag_5dg/benchmark/best.pt novelty_module/data/fnc_quora/train_fnc_processed.txt

### Combining novelty results to get best results on FNC
1) Make sure the paths in the file are right
python novelty_module/novelty_fnc_results_combine.py

## Generating Emotion Results
Download the pre-trained BERT model from
here - (https://github.com/google-research/bert) and unzip them inside the
`bert` directory. In the paper, we use the cased base model.

### Preparing the datasets
python fnc_data_prepare.py
python bytedance_data_prepare.py

1) Training the model with the Klinger dataset
python bert_kling_new.py

- ByteDance dataset
Have to change the path in the code corresponding to the premise, hypothesis files of the train and test datasets (train_ag_dg_hyp.csv and test_ag_dg_hyp.csv)
python bert_classifier_klinger.py

- FNC dataset
Have to change the path in the code corresponding to the premise, hypothesis files of the train and test datasets (train_ag_dg_hyp_fnc.csv and test_ag_dg_hyp_fnc.csv)
python bert_classifier_klinger.py

- Similarly for the other datasets

### Fake Flow Model

Reference - https://aclanthology.org/2021.eacl-main.56/

'''python read_data.py'''

'''python fake_flow.py'''

## Proposed_Model (Folder) -
Contains the implementations of the final proposed model along with the supporting files

## Baselines (Folder) -
Contains the implementations of the baselines along with the supporting files

## Co-Ocurence_Matrices (Folder)
Contains code for generating co-occurance matrices which show the distribution of novelty, emotion and sentiment labels with respect to the ground truth labels.

### References: -

Our code is loosely based on the following   
novelty_module - https://github.com/alibaba-edu/simple-effective-text-matching-pytorch    
emotion_module - https://github.com/google-research/google-research/tree/master/goemotions   
ESIM model - https://github.com/coetaur0/ESIM
