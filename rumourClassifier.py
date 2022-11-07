from transformer import *
from selfTraining import *
import torch
import os
import json, glob, random
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM,BertTokenizer, BertModel

#this is the
model_dict = { 'MBERT': 'bert-base-multilingual-cased',
               'XLMR': 'xlm-roberta-base',
               'BERT': 'bert-base-cased',
               'CHNBERT': 'bert-base-chinese'
               }

tokenizer_dict = {'BERT':'BertTokenizer',
                  'RoBERTa': 'AutoTokenizer'
                }

base_model_dict = { 'MBERT': 'BertModel',
                    'XLMR': 'AutoModelForMaskedLM',
                    'BERT': 'BertModel',
                    'CHNBERT': 'BertModel',
                }



class RumourClassifier(torch.nn.Module):

    def __init__(self,args):
        super(RumourClassifier, self).__init__()
        """
        Text-based rumour classifier
        base_model: define the base multilingual/monolingual transformer encoder
        model_card: encoder version
        do_lower_case: whether do lower case
        """
        self.args = args
        self.gpu = args.gpu
        self.pretrain = args.pretrain
        self.pretrain_path = args.pretrain_data_path
        self.base_model = base_model_dict[args.bert_model]
        self.bert = self.base_model.from_pretrained(model_dict[args.bert_model],num_labels = args.num_labels,output_attentions = False,output_hidden_states = False,return_dict=False)
        self.base_tokenizer = tokenizer_dict[args.bert_model]
        self.lr = args.learning_rate
        self.do_lower_case = args.lower_case
        self.num_labels = args.num_labels
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.num_train_epochs = args.num_train_epochs
        self.warmup_steps = args.warmup_steps
        self.optimizer = None
        self.scheduler = None
        self.linear = nn.Linear(self.bert.config.hidden_size, args.num_labels)
        self.dropout = nn.Dropout(0.2)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=args.num_labels, reduction='sum')
        self.output = '../saved_models/'

    @property
    def device(self):
        if self.gpu:
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def init_base_classifier(self):
        if not self.pretain:
            tokenizer = self.base_tokenizer.from_pretrained(model_card, do_lower_case=self.do_lower_case)
            model = self.bert
        else:
            tokenizer = self.base_tokenizer.from_pretrianed(self.pretrain_path, do_lower_case=self.do_lower_case)
            model = self.base_model.from_pretrained(self.pretrain_path, num_labels = self.num_labels,output_attentions = False,output_hidden_states = False,return_dict=False)
        # Tell pytorch to run this model on the GPU.
        if self.gpu:
            model.cuda()
        return tokenizer, model

    def set_optimizer(self):
        self.otpimizer = AdamW(self.base_model.parameters(),
                              weight_decay = 0.0,
                              lr = self.lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                              eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                            )

    def set_scheduler(self):
        t_total = len(train_dataset) // self.batch_size * self.num_train_epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.warmup_steps,
                                                         num_training_steps= t_total)



    def init_dataloader(self, input_data, test_data):
        train_dataloader, validation_dataloader, prediction_dataloader = init_transformer_dataloader(
            self.base_tokenizer,
            self.model_card,
            self.do_lower_case,
            self.max_length,
            input_data,
            test_data,
            self.batch_size)
        return train_dataloader, validation_dataloader, prediction_dataloader

    def forward(self, dataset):
        a,b,labels = read_text_seq_input(dataset)
        src, seg, mask_src = encode_text_seq(self.base_tokenizer,self.model_card,self.do_lower_case,self.max_length,a,b,labels)
        top_vec, _ = self.bert(input_ids=src, token_type_ids=seg, attention_mask=mask_src)
        top_vec = self.dropout(top_vec)
        top_vec *= mask_src.unsqueeze(dim=-1).float()
        top_vec = torch.sum(top_vec, dim=1) / mask_src.sum(dim=-1).float().unsqueeze(-1)
        preds = self.linear(top_vec).squeeze()
        return preds

    def get_loss(self, dataset):
        output = self.forward(dataset)
        return self.loss(output.view(-1,self.num_labels), label.view(-1))

    def predict(self, dataset):
        output = self.forward(dataset)
        batch_size = output.shape[0]
        prediction = torch.argmax(output, dim=-1).data.cpu().numpy().tolist()
        return prediction

    def finetune_raw(sef, train_data, test_data):
        train_dataloader, validation_dataloader, prediction_dataloader = self.init_dataloader(train_data, test_data)
        optimizer = self.set_optimizer()
        scheduler = self.set_scheduler()
        finetune_stats = transformer_training(self.num_train_epochs, train_dataloader,validation_dataloader, self.bert, optimizer, scheduler)
        return finetune_stats


    def finetune(self, train_dataloader, validation_dataloader):
        optimizer = self.set_optimizer()
        scheduler = self.set_scheduler()
        finetune_stats = transformer_training(self.num_train_epochs, train_dataloader,validation_dataloader, self.bert, optimizer, scheduler)
        return finetune_stats

    def save(self, name):
        file = "{}/{}.pkl".format(self.output, name)
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        logging.info("Saving model to {}".format(name))
        state = {
            "model": self.bert.state_dict()
        }
        torch.save(state, file)

    def apply(self, dataset):
        preds, preds_conf = transformer_predict_only(self.args,dataset)
        return preds, preds_conf

    def train(self, dataset):
        pred_labels, pred_conf = self.apply(self.args,dataset)
        return {
            "preds": pred_labels,
            "conf": pred_conf
        }

    def predict(self, dataset, student_features=None):
        pred_labels, pred_conf = self.apply(dataset)
        return pred_labels, pred_conf




class Student(RumourClassifier):
    pass

class Teacher(RumourClassifier):
    def __init__(self, args):
        Teacher.__init__(self, args)

    def apply(self, dataset):
        preds, preds_conf = transformer_predict_only(self.args,dataset)
        return preds, preds_conf

    def train(self, dataset):
        pred_labels, pred_conf = self.apply(self.args,dataset)
        return {
            "preds": pred_labels,
            "conf": pred_conf
        }

    def predict(self, dataset, student_features=None):
        pred_labels, pred_conf = self.apply(dataset)
        return pred_labels, pred_conf

    def predict_acc(self, dataset):
        pred_labels, pred_conf = self.apply(dataset)
        golds = dataset.labels
        acc = flat_accuracy(pred_labels, golds)
        return pred_labels, pred_conf, acc
