import torch
import torch.nn as nn
import torch.nn.functional as F
import collections 
import logging 
import argparse
import numpy as np
from layers import *
from tools import config_setting

def weightedsum(input_tensor):
        x = input_tensor[0]
        a = input_tensor[1]

        a = torch.unsqueeze(a, -1)
        weighted_input = x * a
        
        return torch.sum(weighted_input, dim=1)


def create_emb_matrix(config, vocab, emb_file_gen, emb_file_domain,):
    """
    Create the emb_matrix according to the vocab and emb_file
    """
    print ('Loading pretrained general word embeddings and domain word embeddings ...')
    embeddings = nn.Embedding(len(vocab), config.emb_dim)
    emb_matrix = embeddings.weight
    emb_matrix.requires_grad = False
    use_domain_emb = config.use_domain_emb

    counter_gen = 0.0
    pretrained_emb = open(emb_file_gen)
    for line in pretrained_emb:
        tokens = line.split()
        if(len(tokens) != 301):
            continue
        word = tokens[0]
        vector = tokens[1:]
        try:
            index = vocab[word]
            t = torch.Tensor(np.array(vector, dtype=float))
            emb_matrix[index, 0:300] = t
            counter_gen += 1
        except KeyError:
            pass
    
    if use_domain_emb:
        counter_domain = 0.0
        pretrained_emb = open(emb_file_domain)
        for line in pretrained_emb:
            tokens = line.split()
            if len(tokens) != 101:
                continue
            word = tokens[0]
            vector = tokens[1:]
            try:
                index = vocab[word]
                t = torch.Tensor(np.array(vector, dtype=float))
                emb_matrix[index, 300:] = t
                counter_domain += 1
            except KeyError:
                pass
    
    pretrained_emb.close()
    print("%i/%i word vectors initialized by general embedding (hit rate: %.2f%%)"%(counter_gen, len(vocab), 100*counter_gen/len(vocab)))
    if use_domain_emb:
        print('%i/%i word vectors initialized by domain embeddings (hit rate: %.2f%%)' % (counter_domain, len(vocab), 100*counter_domain/len(vocab)))


    return emb_matrix


class AE(nn.Module):
    def __init__(self,config,nb_class):
        super(AE,self).__init__()
        self.config = config
        self.cnn = cnn_task(config, config.aspect_layers)
        self.input_dim = (config.shared_layers+1)*config.cnn_dim + config.emb_dim
        self.output_dim = nb_class
        self.dense = dense(self.input_dim, self.output_dim)

    def forward(self,input):
        word_emb = input[0]
        x = input[1]
        x = self.cnn(x)
        x = torch.cat((word_emb, x),dim=-1)
        x = F.dropout(x, self.config.dropout_prob)
        x = self.dense(x)
        return x

class AS(nn.Module):
    def __init__(self, config, use_opinion, overall_maxlen):
        super(AS, self).__init__()
        self.config = config
        self.cnn = cnn_task(config, config.senti_layers+1)
        self.self_attention = self_attention(config, use_opinion,overall_maxlen)
        self.dense = dense(config.cnn_dim*2, 3)

    def forward(self, input):
        '''
        #input: [init_shared-features, sentiment_output, 
        #          op_label_input, aspect_probs, p_gold_op]
        '''
        init_shared_features = input[0]
        x = input[1]
        
        x = self.cnn(x)
        att_input = input[2:]
        att_input.insert(0, x)
        x = self.self_attention(att_input)
        x = torch.cat((init_shared_features, x),-1)
        x = F.dropout(x,self.config.dropout_prob)
        x = self.dense(x)

        return x

class DS(nn.Module):
    def __init__(self,config):
        super(DS, self).__init__()
        self.config = config
        self.cnn = cnn_task(config, config.doc_senti_layers)
        self.attention = attention(config)
        self.dense = dense(config.cnn_dim, 3)

    def forward(self, x, overall_maxlen, phrase=None):
        x = self.cnn(x)
        att_weight_softmax, att_weight_sigmoid = self.attention(x)
        
        senti_weights = torch.unsqueeze(att_weight_sigmoid, dim=-1)
        
        x = weightedsum([x,att_weight_softmax])
        x = F.dropout(x, self.config.dropout_prob)
        x = self.dense(x) # dense层里已经logsoftmax了

        if phrase == 'aspect_model':
            doc_senti_probs = torch.unsqueeze(x, -2)
            doc_senti_probs = torch.repeat_interleave(doc_senti_probs, overall_maxlen, 1)
            return doc_senti_probs, senti_weights
        elif phrase == 'doc_model':
            return x # 已经softmax了

class DD(nn.Module):
    def __init__(self,config):
        super(DD, self).__init__()
        self.config = config
        self.cnn = cnn_task(config, config.doc_domain_layers)
        self.attention = attention(config)
        self.dense = dense(config.cnn_dim, 1, activation='sigmoid')

    def forward(self, x, phrase=None):
        x = self.cnn(x)
        domain_att_weight_softmax, domain_att_weight_sigmoid = self.attention(x)
        
        domain_weights = torch.unsqueeze(domain_att_weight_sigmoid, dim=-1)
        
        x = weightedsum([x,domain_att_weight_softmax])
        x = F.dropout(x, self.config.dropout_prob)
        x = self.dense(x)

        if phrase == 'aspect_model':
            return domain_weights
        elif phrase == 'doc_model':
            return x # sigmoid之后的了
    

class IMN(nn.Module):
    def __init__(self, config, nb_class, use_opinion, overall_maxlen):
        super(IMN, self).__init__()
        self.config = config
        # self.embeddings = embedding_matrix
        self.overall_maxlen = overall_maxlen

        self.CNN = cnn_shared(config)
        self.AE = AE(config, nb_class)
        self.AS = AS(config, use_opinion, overall_maxlen)
        self.DS = DS(config)
        self.DD = DD(config)
        if config.use_doc:
            self.DENSE = dense(config.cnn_dim + nb_class + 8, config.cnn_dim)
        else:
            self.DENSE = dense(config.cnn_dim + nb_class + 3, config.cnn_dim)

    def aspect_model(self, config, input):
        word_emb = input[0]
        sentence_output = input[1]
        init_shared_features = input[1]
        op_label_input = input[2].float()
        p_gold_op = torch.from_numpy(input[3]).float().cuda()

        for i in range(config.interactions + 1):
            aspect_output = sentence_output
            sentiment_output = sentence_output
            # note that the aspet-level data will also go through the doc-level models
            doc_senti_output = sentence_output
            doc_domain_output = sentence_output

            aspect_probs = self.AE([word_emb,aspect_output])
            sentiment_probs = self.AS([init_shared_features, sentiment_output, op_label_input, aspect_probs, p_gold_op])

            if config.use_doc:
                doc_senti_probs, senti_weights = self.DS(doc_senti_output, self.overall_maxlen, phrase='aspect_model')
                domain_weights = self.DD(doc_domain_output, phrase='aspect_model')
                
                sentence_output = torch.cat((sentence_output, aspect_probs, sentiment_probs, \
                                                doc_senti_probs, senti_weights, domain_weights),-1)
            else:
                # update sentence_output for the next iteration
                sentence_output = torch.cat([sentence_output, aspect_probs, sentiment_probs])
            sentence_output = self.DENSE(sentence_output)
        
        return aspect_probs, sentiment_probs

    def doc_model(self,config, input):
        doc_output_1 = input[0]
        doc_output_2 = input[1]

        if config.use_doc:
            doc_prob_1 = self.DS(doc_output_1, self.overall_maxlen, 'doc_model')
            doc_prob_2 = self.DD(doc_output_2, 'doc_model')
            return doc_prob_1, doc_prob_2
        else: 
            return None

    def forward(self, input, phrase):
        '''
        :param  input: aspect_model->[batch_x, batch_y_op, gold_prob]
                       doc_model->[batch_x_1, batch_x_2]
        :param  phrase: 指定模式(aspect_model/doc_model)
        '''
  
        if phrase == 'aspect_model':
            # 先通过共享层
            init_input = []
            for i in range(len(input)):
                if i == 0:
                    word_embeddings,tmp = self.CNN.forward(input[0], phrase)
                    init_input.append(word_embeddings)
                    init_input.append(tmp)
                else: 
                    init_input.append(input[i])
            aspect_probs, sentiment_probs = self.aspect_model(self.config, init_input)
            return aspect_probs, sentiment_probs
        elif phrase == 'doc_model' and self.config.use_doc:
            # 先通过共享层
            init_input = []
            for i in input:
                tmp = self.CNN.forward(i,phrase)
                init_input.append(tmp)

            doc_prob_1, doc_prob_2 = self.doc_model(self.config, init_input)
            return doc_prob_1, doc_prob_2


