import torch
import torch.nn as nn
import torch.nn.functional as F
import collections 
import numpy as np
import logging 
import argparse

class cnn_task(nn.Module):
    def __init__(self,config, num_layers):
        super(cnn_task, self).__init__()
        self.config = config
        self.num_layers = num_layers
        self.cnn_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=self.config.cnn_dim, 
                    out_channels=self.config.cnn_dim,
                    kernel_size=5,padding=2)
            for i in range(num_layers)]
            )
        
        
    def forward(self,x):
        """
        Compute the sentence_output through AE
        :param x:sentence_output whiach size is bs*maxlen*cnn_dim
        """
        for i in range(self.num_layers):
            x = F.dropout(x, self.config.dropout_prob)
            x = x.permute(0,2,1)
            x = F.relu(self.cnn_layers[i](x))
            x = x.permute(0,2,1)
        return x

class cnn_shared(nn.Module):
    def __init__(self, config):
        """
        :param config: all of the arguments
        """
        super(cnn_shared,self).__init__()
        self.config = config
        # create the cnn block
        for i in range(self.config.shared_layers):
            if i == 0:
                self.conv = nn.ModuleList([
                    nn.Conv1d(in_channels=self.config.emb_dim, 
                                out_channels=int(self.config.cnn_dim/2),
                                kernel_size=3, stride=1, padding=1), # len-3+1
                    nn.Conv1d(in_channels=self.config.emb_dim, 
                                out_channels=int(self.config.cnn_dim/2),
                                kernel_size=5, stride=1, padding=2) # len-5+2
                    ])
            else:
                self.conv.append(
                    nn.Conv1d(in_channels=self.config.cnn_dim, 
                                out_channels=self.config.cnn_dim,
                                kernel_size=5,stride=1,padding=2)
                    )


    def forward(self,x,phrase):
        if phrase == 'aspect_model':
            word_embeddings = x
        x = x.permute(0,2,1)

        for i in range(self.config.shared_layers):
            x = F.dropout(x, self.config.dropout_prob)

            if i == 0:
                x1 = F.relu(self.conv[0](x)).permute(0,2,1) # bs*cnn_dim/2*maxlen
                x2 = F.relu(self.conv[1](x)).permute(0,2,1) # bs*cnn_dim/2*maxlen
                # cat in order to keep the size no changing
                x = torch.cat((x1,x2), -1).permute(0,2,1)
            else:
                x = F.relu(self.conv[i+1](x))
            
            if phrase == 'aspect_model':
                word_embeddings = torch.cat((word_embeddings, x.permute(0,2,1)), -1)
        x = x.permute(0,2,1)
        if phrase == 'aspect_model':
            return word_embeddings,x 
        else:
            return x
            
class dense(nn.Module):
    def __init__(self, input_dim, output_dim, activation='softmax'):
        super(dense, self).__init__()
        act_func = {'softmax':nn.LogSoftmax(dim=-1),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU()}
        self.dense_layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            act_func[activation]
        )

    def forward(self,x):
        x = self.dense_layers(x)
        return x

class attention(nn.Module):
    def __init__(self, config, bias=True):
        """
        :param input_shape: bs*maxlen*cnn_dim
        :param bias: whether add bias
        """
        super(attention, self).__init__()
        self.epsilon = torch.tensor((1e-10))
        self.W = nn.Parameter(torch.Tensor(1,config.cnn_dim))
        torch.nn.init.xavier_normal_(self.W)
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.Tensor(1,1))
            torch.nn.init.zeros_(self.b)
            
    
    def forward(self, input, mask=None):
        x = input # bs*maxlen*cnn_dim
        query = torch.unsqueeze(self.W, 0) # 1*1*cnn_dim
        query = self.W

        eij = torch.sum(x*query,-1)
        if self.bias:
            eij = eij + self.b
        a = torch.exp(eij)
        a_sigmoid = torch.sigmoid(eij)

        if mask is not None:
            a = a*mask
            a_sigmoid = a*mask
        
        a = a / torch.sum(a, dim=1, keepdim=True) + self.epsilon

        return [a, a_sigmoid]

class self_attention(nn.Module):
    def __init__(self, config, use_opinion, overall_maxlen, bias=True):
        """
        :params input_shape: bs*maxlen*cnn_dim
        """
        super(self_attention, self).__init__()
        self.epsilon = torch.tensor((1e-10))
        self.use_opinion = use_opinion
        self.bias = bias
        cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.config = config
        self.input_dim = config.cnn_dim
        self.steps = overall_maxlen
        self.W = nn.Parameter(torch.Tensor(self.input_dim, self.input_dim))
        torch.nn.init.xavier_normal_(self.W)
        if bias:
            self.b = nn.Parameter(torch.Tensor(self.input_dim))
            torch.nn.init.zeros_(self.b)
    
    def forward(self, input, mask=None):
        """
        :param input :[sentence_output: bs*maxlen*cnn_dim
                       op_label_input: bs*maxlen*3
                       aspect_probs: bs*maxlen*5
                       p_gold_op: bs*maxlen
                        ]
        :param mask:
        """
        x = input[0]
        gold_opinion = input[1]
        predict_opinion = input[2]
        gold_prob = input[3].cuda(self.device)
        # gold_prob.to(self.device)
        mask = mask

        W = self.W.repeat(self.config.batch_size, 1, 1)
        b = self.b.repeat(self.config.batch_size, self.steps, 1)
        x_tran = torch.bmm(x, W)
        if self.bias:
            x_tran = x_tran + b
        
        x_transpose = x.permute(0,2,1)
        weights = torch.bmm(x_tran, x_transpose)

        # location matrix maxlen*maxlen
        location = np.abs(np.tile(np.array(range(self.steps)), (self.steps,1)) - np.array(range(self.steps)).reshape(self.steps,1))
        location = torch.Tensor(location)
        loc_weights = 1.0 / (location+self.epsilon)
        loc_weights = (loc_weights * (location!=0)).to(self.device)
        weights = weights * loc_weights

        if self.use_opinion:
            gold_opinion_ = gold_opinion[:,:,1]+gold_opinion[:,:,2]
            predict_opinion_ = predict_opinion[:,:,3]+predict_opinion[:,:,4]
            # gold_prob is either 0 or 1 
            opinion_weights = gold_prob*gold_opinion_ + (1.-gold_prob)*predict_opinion_
            opinion_weights = torch.unsqueeze(opinion_weights, dim=-2)
            weights = weights * opinion_weights

        weights = torch.tanh(weights)
        weights = torch.exp(weights)
        weights = weights * torch.Tensor((np.eye(self.steps) == 0)).to(self.device)

        if mask is not None:
            mask = torch.unsqueeze(mak, dim=-2)
            mask = torch.repeat_interleave(mask, self.steps, dim=1)
            weights = weights * mask

        weights = weights / torch.sum(weights, dim=-1, keepdim=True) + self.epsilon

        output = torch.bmm(weights,x)
        return output
