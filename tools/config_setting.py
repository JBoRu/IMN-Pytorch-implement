import argparse
import logging
import numpy as np

def Parse_Arguments():
    parser = argparse.ArgumentParser()

    # argument related to datasets and data preprocessing
    parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='res', help="domain of the corpus {res, lt, res_15}")
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=20000, help="Vocab size. '0' means no limit (default=20000)")

    # hyper-parameters related to network training
    parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")
    parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=80, help="Number of epochs (default=80)")
    parser.add_argument("--validation-ratio", dest="validation_ratio", type=float, metavar='<float>', default=0.2, help="The percentage of training data used for validation")
    parser.add_argument("--pre-epochs", dest="pre_epochs", type=int, metavar='<int>', default=5, help="Number of pretrain document-level epochs (default=5)")
    parser.add_argument("-mr", dest="mr", type=int, metavar='<int>', default=2, help="#aspect-level epochs : #document-level epochs = mr:1")

    # hyper-parameters related to network structure
    parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=400, help="Embeddings dimension (default=dim_general_emb + dim_domain_emb = 400)")
    parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=300, help="CNN output dimension. '0' means no CNN layer (default=300)")
    parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability. (default=0.5)")

    parser.add_argument("--use-doc", dest="use_doc", type=int, metavar='<int>', default=1, help="whether to exploit knowledge from document-level data")
    parser.add_argument("--train-op", dest="train_op", type=int, metavar='<int>', default=1, help="whether to extract opinion terms")
    parser.add_argument("--use-opinion", dest="use_opinion", type=int, metavar='<int>', default=1, help="whether to perform opinion transmission")

    parser.add_argument("--shared-layers", dest="shared_layers", type=int, metavar='<int>', default=2, help="The number of CNN layers in the shared network")
    parser.add_argument("--doc-senti-layers", dest="doc_senti_layers", type=int, metavar='<int>', default=0, help="The number of CNN layers for extracting document-level sentiment features")
    parser.add_argument("--doc-domain-layers", dest="doc_domain_layers", type=int, metavar='<int>', default=0, help="The number of CNN layers for extracting document domain features")
    parser.add_argument("--senti-layers", dest="senti_layers", type=int, metavar='<int>', default=0, help="The number of CNN layers for extracting aspect-level sentiment features")
    parser.add_argument("--aspect-layers", dest="aspect_layers", type=int, metavar='<int>', default=2, help="The number of CNN layers for extracting aspect features")
    parser.add_argument("--interactions", dest="interactions", type=int, metavar='<int>', default=2, help="The number of interactions")
    parser.add_argument("--use-domain-emb", dest="use_domain_emb", type=int, metavar='<int>', default=1, help="whether to use domain-specific embeddings")

    # random seed that affects data splits and parameter intializations
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=123, help="Random seed (default=123)")
    
    # test
    parser.add_argument("--vocab", dest="vocab", default=((1,2,3),(1,2,3)))
    parser.add_argument("--emb_file_gen", dest="emb_file_gen", type=str, default="./corpus/glove/res.txt")
    parser.add_argument("--emb_file_domain", dest="emb_file_domain",type=str, default="./corpus/glove/res.txt")
    args = parser.parse_args()

    return args
