# import ptvsd
# ptvsd.enable_attach(address = ('172.16.71.13', 3000))
# ptvsd.wait_for_attach()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as opt
import collections 
import logging 
import argparse
from modules import *
import tqdm
from sklearn import metrics
from corpus.read_utills import *
from corpus.imn_dataset import ASPECT_Dataset, DOC_Dataset
from tools import config_setting
from torch.nn.utils.rnn import pad_sequence

class Train(object):
    def __init__(self,config, nb_class, overall_maxlen, embedding_matrix, use_opinion, domain, vocab_path):
        # 声明参数
        self.config = config
        # 获得词向量矩阵
        self.embedding_matrix = embedding_matrix
        # 获得单句最大长度
        self.overall_maxlen = overall_maxlen
        # 声明模型
        self.imn_model = IMN(config, nb_class, use_opinion, overall_maxlen)
        
        # 获得可用GPU
        cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        # 将模型发送到GPU/CPU
        self.imn_model.to(self.device)
        # 声明需要的数据集
        train_aspect_dataset = ASPECT_Dataset(config, domain, vocab_path, 'train', overall_maxlen)
        self.train_aspect_dataloader = DataLoader(train_aspect_dataset,
                                                batch_size=config.batch_size,
                                                num_workers=0,
                                                drop_last=True)
        test_aspect_dataset = ASPECT_Dataset(config, domain, vocab_path, 'test', overall_maxlen)
        self.test_aspect_dataloader = DataLoader(test_aspect_dataset,
                                                batch_size=config.batch_size,
                                                num_workers=0,
                                                drop_last=True)
        if config.use_doc:
            doc_senti_dataset = DOC_Dataset(config, vocab_path, 'doc_senti')
            self.doc_senti_dataloader = DataLoader(doc_senti_dataset,
                                                batch_size=config.batch_size,
                                                num_workers=0,
                                                drop_last=True)
            test_doc_senti_dataset = DOC_Dataset(config, vocab_path, 'doc_senti', train=0)
            self.test_doc_senti_dataloader = DataLoader(test_doc_senti_dataset,
                                                batch_size=config.batch_size,
                                                num_workers=0,
                                                drop_last=True)

            doc_domain_dataset = DOC_Dataset(config, vocab_path, 'doc_domain')
            self.doc_domain_dataloader = DataLoader(doc_domain_dataset,
                                                batch_size=config.batch_size,
                                                num_workers=0)
            test_doc_domain_dataset = DOC_Dataset(config, vocab_path, 'doc_domain', train=0)
            self.test_doc_domain_dataloader = DataLoader(test_doc_domain_dataset,
                                                batch_size=config.batch_size,
                                                num_workers=0)

        # 声明模型需要优化的参数
        self.optim_params = list(self.imn_model.parameters())
        # 声明优化器
        self.optimizer = self.get_optimizer(config)

    def get_optimizer(self,config):
        # if config.algorithm == 'sgd':
        return opt.SGD(self.optim_params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False)
    
    def get_prob(self,epoch_count):
        prob = 5/(5+np.exp(epoch_count/5))
        return prob

    def pre_train(self, epoch):
        self.imn_model.train()
        self.iteration(epoch, [self.doc_senti_dataloader, self.doc_domain_dataloader], 'pretrain', train=True)
    
    def pre_test(self, epoch):
        self.imn_model.eval()
        with torch.no_grad():
            self.iteration(epoch, [self.test_doc_senti_dataloader, self.test_doc_domain_dataloader], 'pretrain', train=False)

    def train(self, epoch):
        self.imn_model.train()
        self.iteration(epoch, [self.train_aspect_dataloader, self.doc_senti_dataloader, self.doc_domain_dataloader], 'train', train=True)
    
    def test(self, epoch):
        self.imn_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, [self.test_aspect_dataloader], 'train', train=False)

    def iteration(self, epoch, data_loader, phrase, train=True):
        '''
        :param phrase: 控制模型的训练阶段（pretrain/train）
        :param train: 控制模型是训练还是运行阶段 （True/False）
        '''
        if phrase == 'pretrain':
            senti_dl = data_loader[0]
            domain_dl = data_loader[1] 
            str_code = 'train' if train else 'test'
            data_iter = tqdm.tqdm(enumerate(zip(senti_dl, domain_dl)),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

            total_senti_loss = 0
            total_domain_loss = 0 
            total_loss = 0
            all_senti_pre, all_senti_lab = [], []
            all_domain_pre, all_domain_lab = [], []
            
            bs = 0 # 计数batc_size
            for i, d in data_iter:
                # 取得数据并发送给计算设备
                senti_data = d[0]
                domain_data = d[1]
                batch_senti_x = senti_data['x'].to(self.device)
                batch_senti_y = senti_data['y'].to(self.device)
                batch_domain_x = domain_data['x'].to(self.device)
                batch_domain_y = domain_data['y'].to(self.device)
                # 转换为对应的词向量
                bt_senti_x_emb = self.embedding_matrix[batch_senti_x].cuda()
                bt_domain_x_emb = self.embedding_matrix[batch_domain_x].cuda()
                # 前向传播得到结果
                doc_prob_1, doc_prob_2 = self.imn_model.forward([bt_senti_x_emb, bt_domain_x_emb], 'doc_model')
                if train:
                    # 计算损失
                    loss1 = self.compute_loss(doc_prob_1, batch_senti_y, 'NLLLoss')
                    loss2 = self.compute_loss(doc_prob_2, batch_domain_y, 'BCELoss')
                    loss = loss1 + loss2
                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # 为计算当前epoch的平均loss
                    total_senti_loss += loss1.item()
                    total_domain_loss += loss2.item()
                    total_loss += loss.item()
                else:
                    # 提取出预测的结果和标记，并存在all_predictions, all_label里
                    doc_prob_1_ = doc_prob_1.argmax(dim=-1).tolist()
                    senti_y = batch_senti_y.cpu().argmax(dim=-1).tolist()
                    all_senti_pre.extend(doc_prob_1_)
                    all_senti_lab.extend(senti_y)

                    doc_prob_2_ = doc_prob_2.squeeze().tolist()
                    doc_prob_2_ = [1 if i >= 0.5 else 0 for i in doc_prob_2_]
                    domain_y = batch_domain_y.cpu().squeeze().tolist()
                    all_domain_pre.extend(doc_prob_2_)
                    all_domain_lab.extend(domain_y)

                bs += 1

            # 计算auc
            senti_auc = metrics.recall_score(all_senti_lab, all_senti_lab, average='micro')
            domain_auc = metrics.accuracy_score(all_domain_lab, all_domain_pre)
                
            # 打印输出
            if train:
                print("Pretrain doc-level model: Epoch: %d, senti_loss: %f, domain_loss: %f, loss: %f"%(epoch, total_senti_loss/(bs), total_domain_loss/(bs), total_loss/(bs)))
            else:
                print("Pretest doc-level model: Epoch: %d, senti_auc: %f, domain_auc: %f"%(epoch, senti_auc, domain_auc))

        elif phrase == 'train':
            if train:
                # 训练阶段
                aspect_dl = data_loader[0]
                senti_dl = data_loader[1]
                senti_dl_iter = iter(senti_dl)
                domain_dl = data_loader[2]
                domain_dl_iter = iter(domain_dl)

                str_code = 'train'
                # data_iter = tqdm.tqdm(enumerate(aspect_dl),
                #                 desc="EP_%s:%d \n" % (str_code, epoch),
                #                 total=len(aspect_dl),
                #                 bar_format="{l_bar}{r_bar}")
                data_iter = tqdm.tqdm(enumerate(aspect_dl))

                gold_prob = self.get_prob(epoch)
                rnd = np.random.uniform()
                if rnd < gold_prob:
                    gold_prob = np.ones((self.config.batch_size, self.overall_maxlen))
                else:
                    gold_prob = np.zeros((self.config.batch_size, self.overall_maxlen))

                total_loss, total_aspect_loss, total_senti_loss = 0, 0, 0

                # 记录有多少个batch
                bs = 0
                for i, data in data_iter:  # 一个batch
                    batch_x = data['x']
                    batch_y_ae = data['y_aspect'].to(self.device)
                    batch_y_as = data['y_sentiment'].to(self.device)
                    batch_y_op = data['y_opinion'].to(self.device)
                    batch_mask = data['y_mask'].to(self.device)
                    
                    # 转换为词向量
                    bt_x_emb = self.embedding_matrix[batch_x].cuda()
                    # 前向传播
                    aspect_probs, sentiment_probs = self.imn_model.forward([bt_x_emb, batch_y_op, gold_prob], 'aspect_model')
                    
                    # aspect_probs = aspect_probs.view([self.config.batch_size*self.overall_maxlen, self.nb_class])
                    # sentiment_probs = sentiment_probs.view([self.config.batch_size*self.overall_maxlen, self.nb_class])
                    # 计算损失
                    aspect_probs = aspect_probs.permute(0,2,1)
                    sentiment_probs = sentiment_probs.permute(0,2,1)
                    aspect_loss = self.compute_loss(aspect_probs, batch_y_ae,'NLLLoss')
                    senti_loss = self.compute_loss(sentiment_probs, batch_y_as, 'NLLLoss')
                    loss = aspect_loss + senti_loss
                    # 清空梯度，反向传播
                    # 在更新aspect model时需要注意固定住DS和DD任务相关的层
                    if self.config.use_doc == 1 and self.config.interactions > 0:
                    # fix the document-specific parameters when updating aspect model
                        for name, param in self.imn_model.named_parameters():
                            if 'DS' in name or 'DD' in name:
                                param.requires_grad = False

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # 累计loss
                    total_loss += loss.item()
                    total_aspect_loss += aspect_loss.item()
                    total_senti_loss += senti_loss.item()

                    # 复原以便在交替训练时可以更新DS/DD任务参数
                    if self.config.use_doc == 1 and self.config.interactions > 0:
                    # allow the document-specific parameters when updating doc model
                        for name, param in self.imn_model.named_parameters():
                            if 'DS' in name or 'DD' in name:
                                param.requires_grad = True
                    
                    # 插入doc训练
                    if i%self.config.mr == 0 and self.config.use_doc:
                        senti_data = next(senti_dl_iter)
                        domain_data = next(domain_dl_iter)
                        x_1, y_1 = senti_data['x'], senti_data['y']
                        x_2, y_2 = domain_data['x'], domain_data['y']

                        x_1_emb = self.embedding_matrix[x_1].cuda()
                        y_1 = y_1.to(self.device)

                        x_2_emb = self.embedding_matrix[x_2].cuda()
                        y_2 = y_2.to(self.device)

                        doc_prob_1, doc_prob_2 = self.imn_model.forward([x_1_emb, x_2_emb], 'doc_model')
                        
                        loss1 = self.compute_loss(doc_prob_1, y_1, 'NLLLoss')
                        loss2 = self.compute_loss(doc_prob_2, y_2, 'BCELoss')
                        loss = loss1 + loss2
                        # 反向传播
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    bs += 1
                
                # 一个epoch结束
                av_loss = total_loss/bs
                av_as_loss = total_aspect_loss/bs
                av_sen_loss = total_senti_loss/bs
                print('aspect-level model train: Epoch %d, loss: %f, as_loss: %f, sen_loss: %f' % (epoch, av_loss, av_as_loss, av_sen_loss))
            else:
                # 测试阶段
                aspect_dl = data_loader[0]

                all_aspect_lab, all_aspect_pre, all_senti_lab, all_senti_pre, all_mask = [], [], [], [], []
                total_loss, total_aspect_loss, total_senti_loss = 0, 0, 0
                
                bs = 0
                # 遍历所有数据
                for i, data in enumerate(aspect_dl):
                    batch_x = data['x']
                    batch_y_op = data['y_opinion'].to(self.device)
                    batch_mask = data['y_mask'].to(self.device)
                    
                    batch_y_ae = data['y_aspect'].to(self.device)
                    batch_y_as = data['y_sentiment'].to(self.device)

                    bt_x_emb = self.embedding_matrix[batch_x].cuda()
                    # 测试时直接使用预测的opinion信息
                    batch_gold_prob = np.zeros((batch_x.size()[0], self.overall_maxlen))

                    # 测试数据
                    aspect_probs, sentiment_probs = self.imn_model.forward([bt_x_emb, batch_y_op, batch_gold_prob],phrase='aspect_model')
                    
                    # 对测试集仍然采用损失函数测试
                    aspect_probs = aspect_probs.permute(0,2,1)
                    sentiment_probs = sentiment_probs.permute(0,2,1)
                    aspect_loss = self.compute_loss(aspect_probs, batch_y_ae,'NLLLoss')
                    senti_loss = self.compute_loss(sentiment_probs, batch_y_as, 'NLLLoss')
                    loss = aspect_loss + senti_loss
                    # 累计loss
                    total_loss += loss.item()
                    total_aspect_loss += aspect_loss.item()
                    total_senti_loss += senti_loss.item()
                    
                    # 计算auc
                    all_aspect_lab.extend(batch_y_ae)
                    all_senti_lab.extend(batch_y_as)
                    all_aspect_pre.extend(aspect_probs)
                    all_senti_pre.extend(sentiment_probs)
                    all_mask.extend(batch_mask)
                    
                    bs += 1
                all_aspect_lab = [_.tolist() for _ in all_aspect_lab]
                all_senti_lab = [_.tolist() for _ in all_senti_lab]
                all_aspect_pre = [_.tolist() for _ in all_aspect_pre]
                all_senti_pre = [_.tolist() for _ in all_senti_pre]
                all_mask = [_.tolist() for _ in all_mask]

                # 计算得分还有问题！！
                # f_aspect, f_opinion, acc_s, f_senti, f_absa \
                    # = self.get_metric(all_aspect_lab, all_aspect_pre, all_senti_lab, all_senti_pre, all_mask, self.config.train_op)
                
                # print('Train aspect-level model: Epoch %d, f_aspect: %f, f_opinion: %f, f_senti: %f, f_absa: %f' % (epoch, f_aspect, f_opinion, f_senti, f_absa))
                # 一个epoch结束
                av_loss = total_loss/bs
                av_as_loss = total_aspect_loss/bs
                av_sen_loss = total_senti_loss/bs
                print('aspect-level model test: Epoch %d, loss: %f, as_loss: %f, sen_loss: %f' % (epoch, av_loss, av_as_loss, av_sen_loss))
                # return f_absa
                return av_loss

    def convert_to_list(self, y_aspect, y_sentiment, mask):
        y_aspect_list = [] # 所有sentence的label构成的列表
        y_sentiment_list = []
        # 取出每个句子和该句子对应的mask
        for seq_aspect, seq_sentiment, seq_mask in zip(y_aspect, y_sentiment, mask):
            l_a = [] # 一个sentence的每个字对应的aspect_label 
            l_s = [] # 一个sentence的每个字对应的sentiment_label
            # 取出每个字和该字对应的mask(该字是否是padding的)
            for label_dist_a, label_dist_s, m in zip(seq_aspect, seq_sentiment, seq_mask):
                if m == 0: # 是pandding，就不算
                    break
                else:
                    # 对一个字的 aspect_label one-hot表示取argmax可以得到该字的label，例如1，2，3
                    l_a.append(np.argmax(label_dist_a))
                    ### all entries are zeros means that it is a background word or word with conflict sentiment
                    ### which are not counted for training AS
                    ### also when evaluating, we do not count conflict examples
                    # 对一个字的 sentiment_label one-hot表示，如果全为0，说明是一个背景词或者中性词，不考虑
                    if not np.any(label_dist_s):
                        l_s.append(0)
                    else:
                        l_s.append(np.argmax(label_dist_s)+1)
            y_aspect_list.append(l_a)
            y_sentiment_list.append(l_s)
        return y_aspect_list, y_sentiment_list

    def score(self, true_aspect, predict_aspect, true_sentiment, predict_sentiment, train_op):
        if train_op:
            begin = 3
            inside = 4
        else:
            begin = 1
            inside = 2

            # predicted sentiment distribution for aspect terms that are correctly extracted
            pred_count = {'pos':0, 'neg':0, 'neu':0}
            # gold sentiment distribution for aspect terms that are correctly extracted
            rel_count = {'pos':0, 'neg':0, 'neu':0}
            # sentiment distribution for terms that get both span and sentiment predicted correctly
            correct_count = {'pos':0, 'neg':0, 'neu':0}
            # sentiment distribution in original data
            total_count = {'pos':0, 'neg':0, 'neu':0}

            polarity_map = {1: 'pos', 2: 'neg', 3: 'neu'}

            # count of predicted conflict aspect term
            predicted_conf = 0

        correct, predicted, relevant = 0, 0, 0

        for i in range(len(true_aspect)):
            true_seq = true_aspect[i]
            predict = predict_aspect[i]
            
            # 遍历句子的每一个字
            for num in range(len(true_seq)):
                if true_seq[num] == begin:
                    relevant += 1
                    if not train_op:
                        if true_sentiment[i][num]!=0:
                            total_count[polarity_map[true_sentiment[i][num]]]+=1
                        
                    if predict[num] == begin:# 预测的aspect的起始位置是正确的
                        match = True 
                        # 判断预测与真实aspect后续是否完全匹配正确
                        for j in range(num+1, len(true_seq)):
                            if true_seq[j] == inside and predict[j] == inside:
                                continue
                            elif true_seq[j] != inside  and predict[j] != inside:
                                break
                            else:
                                match = False
                                break
                        # 如果后续完全匹配正确，说明预测的aspect是准确的
                        if match:
                            correct += 1
                            if not train_op: # 不抽取opinion
                                # do not count conflict examples
                                if true_sentiment[i][num]!=0:
                                    rel_count[polarity_map[true_sentiment[i][num]]]+=1
                                    pred_count[polarity_map[predict_sentiment[i][num]]]+=1
                                    if true_sentiment[i][num] == predict_sentiment[i][num]:
                                        # aspect的范围和极性都匹配 前提是对一个aspect的所有词极性评价是一致的
                                        correct_count[polarity_map[true_sentiment[i][num]]]+=1

                                else:
                                    predicted_conf += 1

            for pred in predict:
                if pred == begin:
                    predicted += 1

        p_aspect = float(correct / (predicted + 1e-6))
        r_aspect = float(correct / (relevant + 1e-6))
        # F1 score for aspect (opinion) extraction
        f_aspect = float(2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6))

        acc_s, f_s, f_absa = 0, 0, 0

        if not train_op:
            num_correct_overall = correct_count['pos']+correct_count['neg']+correct_count['neu']
            num_correct_aspect = rel_count['pos']+rel_count['neg']+rel_count['neu']
            num_total = total_count['pos']+total_count['neg']+total_count['neu']

            acc_s = num_correct_overall/(num_correct_aspect+1e-6)
        
            p_pos = correct_count['pos'] / (pred_count['pos']+1e-6)
            r_pos = correct_count['pos'] / (rel_count['pos']+1e-6)
            
            p_neg = correct_count['neg'] / (pred_count['neg']+1e-6)
            r_neg = correct_count['neg'] / (rel_count['neg']+1e-6)

            p_neu = correct_count['neu'] / (pred_count['neu']+1e-6)
            r_neu= correct_count['neu'] / (rel_count['neu']+1e-6)

            pr_s = (p_pos+p_neg+p_neu)/3.0
            re_s = (r_pos+r_neg+r_neu)/3.0
            # F1 score for AS only
            f_s = 2*pr_s*re_s/(pr_s+re_s+1e-6) # 出现分母为0！！！
         
            precision_absa = num_correct_overall/(predicted+1e-6 - predicted_conf)
            recall_absa = num_correct_overall/(num_total+1e-6)
            # F1 score of the end-to-end task
            f_absa = 2*precision_absa*recall_absa/(precision_absa+recall_absa+1e-6)

        return f_aspect, acc_s, f_s, f_absa

    def get_metric(self, y_true_aspect, y_predict_aspect, y_true_sentiment, y_predict_sentiment, mask, train_op):
        f_a, f_o = 0, 0
        true_aspect, true_sentiment = self.convert_to_list(y_true_aspect, y_true_sentiment, mask)
        predict_aspect, predict_sentiment = self.convert_to_list(y_predict_aspect, y_predict_sentiment, mask)

        f_aspect, acc_s, f_s, f_absa = self.score(true_aspect, predict_aspect, true_sentiment, predict_sentiment, 0)

        if train_op:
            f_opinion, _, _, _ = self.score(true_aspect, predict_aspect, true_sentiment, predict_sentiment, 1)

        return f_aspect, f_opinion, acc_s, f_s, f_absa

    def compute_loss(self, input, label, function):
        '''
        :param function:使用的损失函数名称
                        ['NLLLoss','BCELoss']
        '''
        if function == 'NLLLoss': # 多分类并且已对结果分布进行softmax
            loss = nn.NLLLoss()
            label = torch.max(label,dim=-1)[1]
        elif function == 'BCELoss': # 二分类
            loss = nn.BCELoss() 
            label = label.float()

        l = loss(input, label)
        return l
        

###test###
if __name__ == "__main__":
    
    domain = 'res'
    config = config_setting.Parse_Arguments()
    nb_class = 5
    vocab_path = './corpus/data_preprocessed/%s/vocab'%domain
    vocab = load_vocab(vocab_path)
    overall_maxlen = prepare_data(domain, config.vocab_size, config.use_doc)
    print(overall_maxlen)
    # overall_maxlen = 78
    if config.use_doc:
        emb_path_gen = './corpus/glove/%s_.txt'%(config.domain)
        emb_path_domain = './corpus/domain_specific_emb/%s_.txt'%(config.domain)
    else:
        emb_path_gen = './corpus/glove/%s.txt'%(config.domain)
        emb_path_domain = './corpus/domain_specific_emb/%s.txt'%(config.domain)
    
    embedding_matrix = create_emb_matrix(config, vocab, emb_path_gen, emb_path_domain)


    trainer = Train(config, nb_class, overall_maxlen, embedding_matrix, config.use_opinion, domain, vocab_path)

    # 预训练doc-level task
    if config.use_doc:
        for ii in range(config.pre_epochs):
            trainer.pre_train(ii)
            trainer.pre_test(ii)

    # 训练aspect-level task
    best_test_f1 = 0
    save_model = False
    for ii in range(config.epochs):
        trainer.train(ii)
        f1 = trainer.test(ii)
        if f1 > best_test_f1 and ii > 60:
            best_test_f1 = f1
            save_model = True
        else:
            save_model = False
        
        if save_model:
            torch.save(trainer.imn_model.state_dict, 'ep_%d_f1_%f_model.pth'%(ii,best_test_f1))
            print ('-------------- Save model --------------')
            break