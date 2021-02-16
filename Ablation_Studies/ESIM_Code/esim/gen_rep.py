"""
Test the ESIM model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

import time
import pickle
import argparse
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from data import NLIDataset
from model import ESIM
from utils import correct_predictions
from torch.nn.parameter import Parameter


def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.

    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.

    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0

    # Deactivate autograd for evaluation.
    pred_file = open('pred_details.txt', 'w')
    rep_file = open('pre_logits_details.txt', 'w')
    save_rep = dict()
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_start = time.time()
            print(i)

            # Move input and output data to the GPU if one is used.
            run_id = batch["id"]
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            #labels = batch["label"].to(device)

            pre_logits, logits, probs = model(premises,
                                         premises_lengths,
                                         hypotheses,
                                         hypotheses_lengths)
            # np_run_id = run_id.cpu().numpy()
            np_pre_logits = pre_logits.cpu().numpy()
            save_rep[run_id.item()] = np_pre_logits

            # Analyze pre_logits and run_id
            if i < 5:
                print('pre_logits is '+str(np_pre_logits), file = rep_file)
                print('run_id is '+str(run_id.item()), file=rep_file)

            #accuracy += correct_predictions(probs, labels)
            #_, predict_label = correct_predictions(probs, labels)
            predict_value, predict_label = probs.max(dim=1)
            print(i, 'predict_label is', predict_label, file = pred_file)
            predictions.append(predict_label.item())
            batch_time += time.time() - batch_start

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    #accuracy /= (len(dataloader.dataset))

    return batch_time, total_time, predictions, save_rep


def main(test_file, pretrained_file, embeddings_file, batch_size=1):
    """
    Test the ESIM model with pretrained weights on some dataset.

    Args:
        test_file: The path to a file containing preprocessed NLI data.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        vocab_size: The number of words in the vocabulary of the model
            being tested.
        embedding_dim: The size of the embeddings in the model.
        hidden_size: The size of the hidden layers in the model. Must match
            the size used during training. Defaults to 300.
        num_classes: The number of classes in the output of the model. Must
            match the value used during training. Defaults to 3.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    debug_file = open('debug.txt', 'w')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    print(20 * "=", " Preparing for generating representations ", 20 * "=")

    checkpoint = torch.load(pretrained_file, map_location = "cpu")

    # Retrieving model parameters from checkpoint.
    vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
    embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
    hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
    num_classes = checkpoint["model"]["_classification.4.weight"].size(0)

    print("\t* Loading the data...")
    with open(test_file, "rb") as pkl:
        test_data = NLIDataset(pickle.load(pkl), max_premise_length = 40, max_hypothesis_length = 15)
    print(test_data, file=debug_file)

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("\t* Building model...")

    # loading the embedding weights separately
    # with open(embeddings_file, "rb") as pkl:
    pkl = open(embeddings_file, "rb")
    embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)\
                 .to(device)
    pkl.close()

    # model = ESIM(vocab_size,
    #              embedding_dim,
    #              hidden_size,
    #              num_classes=num_classes,
    #              device=device).to(device)
    model = ESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 num_classes=num_classes,
                 device=device).to(device)
    # Writing custom load_state_dict
    pretrained_dict = checkpoint["model"]
    own_state = model.state_dict()
    for i, (name, param) in enumerate(pretrained_dict.items()):
        #print(name, type(name))
        # if name is "_word_embedding.weight":
        #     print(name)
        #     continue
        if i==0:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

    #model.load_state_dict(checkpoint["model"])

    print(20 * "=",
          " Loading the representations from ESIM model on device: {} ".format(device),
          20 * "=")
    batch_time, total_time, predictions, save_rep = test(model, test_loader)
    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s,%".format(batch_time, total_time))
    file_debug = open('save_rep_details.txt', 'w')
    print('len of save_rep is'+str(len(save_rep)), file = file_debug)
    try:
        print('save_rep sample key is'+str(list(save_rep.keys())[0]), file = file_debug)
        print('save_rep sample value is'+str(list(save_rep.values())[0]), file = file_debug)
    except:
        pass
    file_debug.close()
    file_debug = open('labels_details.txt', 'a+')
    print('len of predictions is'+str(len(predictions)), file = file_debug)

    # Dumping these predictions as a csv
    df_labels = pd.DataFrame(predictions)
    df_labels.to_csv(test_file.split('.')[0]+'_labels.csv')

    # Dump save_rep as a pickle file
    with open(test_file.split('.')[0]+'_enc.pickle', 'wb') as handle:
        pickle.dump(save_rep, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the ESIM model on\
 some dataset")
    parser.add_argument("--test_data",
                        help="Path to a file containing preprocessed test data")
    parser.add_argument("--checkpoint",
                        help="Path to a checkpoint with a pretrained model")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size to use during testing")
    parser.add_argument("--embeddings_file",
                        help="Path to embeddings file of respective datasets")
    args = parser.parse_args()

    #embeddings_file = '../data/preprocessed_fnc/embeddings.pkl'

    main(args.test_data,
         args.checkpoint,
         args.embeddings_file,
         args.batch_size)
