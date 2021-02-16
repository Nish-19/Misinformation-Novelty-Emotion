This Folder contains the ESIM code used for one of the ablation studies.

Instructions: -
1) Place the glove.300d emebeddings file in data/embeddings
2) Download the best model checkpoint from https://github.com/coetaur0/ESIM/tree/master/data/checkpoints/SNLI and place it in the data/checkpoint/snli directory

Preprocess the ByteDance and FNC dataset - 
1) python \scripts\preprocessing\preprocess_bytedance.py
2) python \scripts\preprocessing\preprocess_fnc.py

To run the Predictions of ESIM model: -
1) cd esim

For ByteDance dataset:-

Run the gen_rep.py file as follows
python gen_rep.py --test_data ../data/preprocessed_bytedance/SNLI/train_data.pkl --checkpoint ../data/checkpoints/SNLI/best.pth --embeddings_file ../data/preprocessed_bytedance/SNLI/embeddings.pkl

Please run this by changing train_data.pkl to test_data.pkl for testset

For FNC dataset: -

Run the gen_rep.py file as follows
python gen_rep.py --test_data ../data/preprocessed_fnc/train_data.pkl --checkpoint ../data/checkpoints/SNLI/best.pth --embeddings_file ../data/preprocessed_fnc/embeddings.pkl

Please run this by changing train_data.pkl to test_data.pkl for testset

This will generate the representations as well as the labels that will be used for final classification.