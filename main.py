
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
import os
import numpy as np
import itertools

from mydataloader import MyDataloader
from params import Params

class Main(object):

    def __init__(self):
        pass

    @staticmethod
    def load_dataset():
        data = pd.read_excel(Params.dataset_dir)
        return data

    def do_lowercase(data):
        data["text"] = data["text"].map(lambda x: str(x).lower())
        data["label"] = data["label"].map(lambda x: str(x).lower())
        return data

    @staticmethod
    def find_max_mean_min_sentence_size(X_train):
        df = pd.DataFrame(X_train, columns=["text"])
        df["text_size"] = df["text"].apply(lambda x: len(str(x).split()))
        max_sentence_size = df["text_size"].max()
        mean_sentence_size = int(df["text_size"].mean())
        min_sentence_size = df["text_size"].min()
        Params.max_sentence_size = 100 # mean_sentence_size
        print("max_sentence_size (on X_train): ", max_sentence_size)
        print("mean_sentence_size (on X_train): ", mean_sentence_size)
        print("min_sentence_size (on X_train): ", min_sentence_size)

    @staticmethod
    def find_unique_labels(y_train):
        df = pd.DataFrame(y_train, columns=["label"])
        Params.label2id = {label: id for id, label in enumerate(list(set(df["label"].values)))}
        print(Params.label2id)

    @staticmethod
    def do_preprocessing(data):
        data = Main.do_lowercase(data)
        return data

    @staticmethod
    def get_dataloaders(data):
        # preprocess
        data = Main.do_preprocessing(data)

        X_train, X_test, y_train, y_test = train_test_split(data["text"].tolist(),
                                                            data["label"].tolist(),
                                                            test_size=Params.test_split_rate,
                                                            stratify=data["label"].tolist(),
                                                            shuffle=True,
                                                            random_state=42)

        # find max/mean/min sentence size on X_train
        Main.find_max_mean_min_sentence_size(X_train)

        # find unique labels
        Main.find_unique_labels(y_train)

        # create dataloaders
        train_dataloader = DataLoader(dataset=MyDataloader(X_train, y_train),
                                      batch_size=Params.batch_size,
                                      shuffle=Params.batch_shuffle)
        test_dataloader = DataLoader(dataset=MyDataloader(X_test, y_test),
                                     batch_size=Params.batch_size,
                                     shuffle=Params.batch_shuffle)

        return train_dataloader, test_dataloader

    @staticmethod
    def prepare_batch(X, y):
        encoding = Params.tokenizer(list(X),
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
                                    max_length=Params.max_sentence_size,
                                    add_special_tokens=True)
        # print("encoding:", encoding)
        input_ids = encoding['input_ids']
        token_type_ids = encoding["token_type_ids"]
        attention_mask = encoding['attention_mask']

        # convert tensors to variables
        input_ids = Variable(input_ids, requires_grad=False)
        token_type_ids = Variable(token_type_ids, requires_grad=False)
        attention_mask = Variable(attention_mask, requires_grad=False)
        y = Variable(y.long(), requires_grad=False)  # y: long format

        _tuple = (input_ids, token_type_ids, attention_mask, y)
        return _tuple

    @staticmethod
    def calculate_metrics(logits, y, mode="train"):
        if mode == "train":
            print("--classification_report (train)--")
        elif mode == "eval":
            print("--classification_report (eval)--")
        target_names = [label for label, id in Params.label2id.items()]
        print(classification_report(y, logits, target_names=target_names))
        f1_micro = f1_score(y, logits, average='micro')
        f1_macro = f1_score(y, logits, average='macro')
        f1_weighted = f1_score(y, logits, average='weighted')
        return f1_micro, f1_macro, f1_weighted

    @staticmethod
    def run_train(dataloader):
        Params.model.train()  # set train mode
        epoch_loss = []
        predicted_labels = []
        y_labels = []
        for id, (X, y) in enumerate(dataloader):  # batch
            print("train batch id: ", id)
            (input_ids, token_type_ids, attention_mask, y) = Main.prepare_batch(X, y)

            if Params.use_cuda:
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                y = y.cuda()

            loss, logits = Params.model(input_ids, token_type_ids, attention_mask, labels=y)
            epoch_loss.append(loss.item())  # save batch loss

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()  # (32, 7) -> (batch_size, label_size)
            labels = y.to('cpu').numpy()  # (32,)
            logits_flat = np.argmax(logits, axis=1).flatten()  # (32,)
            labels_flat = labels.flatten()  # (32,)
            predicted_labels.append(logits_flat)
            y_labels.append(labels_flat)

            # backward and optimize
            Params.optimizer.zero_grad()
            Params.model.zero_grad()
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(Params.model.parameters(), Params.max_grad_norm)
            Params.optimizer.step()
            Params.scheduler.step()

            # empty_cache
            torch.cuda.empty_cache()

        # calculate metrics
        pred_flat = list(itertools.chain(*predicted_labels)) # flat list of list to list
        y_flat = list(itertools.chain(*y_labels))
        f1_micro, f1_macro, f1_weighted = Main.calculate_metrics(pred_flat, y_flat, mode="train")
        return mean(epoch_loss), f1_micro, f1_macro, f1_weighted

    @staticmethod
    def run_test(dataloader):
        Params.model.eval()  # set eval mode
        epoch_loss = []
        predicted_labels = []
        y_labels = []
        for id, (X, y) in enumerate(dataloader):  # batch
            print("test batch id: ", id)
            (input_ids, token_type_ids, attention_mask, y) = Main.prepare_batch(X, y)

            if Params.use_cuda:
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                y = y.cuda()

            loss, logits = Params.model(input_ids, token_type_ids, attention_mask, labels=y)
            epoch_loss.append(loss.item())  # save batch loss

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()  # (32, 7) -> (batch_size, label_size)
            labels = y.to('cpu').numpy()  # (32,)
            logits_flat = np.argmax(logits, axis=1).flatten()  # (32,)
            labels_flat = labels.flatten()  # (32,)
            predicted_labels.append(logits_flat)
            y_labels.append(labels_flat)

            # empty_cache
            torch.cuda.empty_cache()

        # calculate metrics
        pred_flat = list(itertools.chain(*predicted_labels))  # flat list of list to list
        y_flat = list(itertools.chain(*y_labels))
        f1_micro, f1_macro, f1_weighted = Main.calculate_metrics(pred_flat, y_flat, mode="eval")
        return mean(epoch_loss), f1_micro, f1_macro, f1_weighted

    @staticmethod
    def get_metrics(y_pred, y_true):
        acc = accuracy_score(y_true, y_pred)
        return acc

    @staticmethod
    def plot_loss(train_loss, test_loss):
        # loss figure
        plt.clf()
        plt.plot(train_loss, label='train')
        plt.plot(test_loss, label='valid')
        plt.title('train-valid loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        # plt.show()
        plt.savefig(os.path.join(Params.plot_dir, "loss.png"))

    @staticmethod
    def plot_f1(train_f1, test_f1, mode="micro"):
        # f1 figure
        plt.clf()
        plt.plot(train_f1, label='train')
        plt.plot(test_f1, label='valid')
        plt.title('train-valid f1-'+mode)
        plt.ylabel('f1-'+mode)
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(Params.plot_dir, "f1-"+mode+".png"))

    @staticmethod
    def create_optimizer():
        if Params.FULL_FINETUNING:
            param_optimizer = list(Params.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(Params.model.classifier.named_parameters()) # "classifier": last layer name
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        Params.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=Params.AdamW_lr,
            eps=Params.AdamW_eps,
            weight_decay=Params.weight_decay
        )

    @staticmethod
    def create_scheduler(train_dataloader):
        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * Params.epoch

        # Create the learning rate scheduler.
        Params.scheduler = get_linear_schedule_with_warmup(
            Params.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

    @staticmethod
    def run_train_test(train_dataloader, test_dataloader):
        # create tokenizer
        Params.tokenizer = BertTokenizer.from_pretrained(Params.bert_model_name)

        # create model

        # create from model name, uses default config
        Params.model = BertForSequenceClassification.from_pretrained(Params.bert_model_name,
                                                                     num_labels=len(Params.label2id),
                                                                     output_attentions=False,
                                                                     output_hidden_states=False)

        print("---model parameters---")
        for name, module in Params.model.named_modules():
            print(name)
            print(module)
            print()

        # push model to gpu
        if Params.use_cuda:
            Params.model.cuda()
        model_total_params = sum(p.numel() for p in Params.model.parameters())
        print("model_total_params: ", model_total_params)

        # create optimizer
        Main.create_optimizer()

        # create schedular
        Main.create_scheduler(train_dataloader)

        train_loss = []
        train_f1_micro = []
        train_f1_macro = []
        train_f1_weighted = []

        test_loss = []
        test_f1_micro = []
        test_f1_macro = []
        test_f1_weighted = []

        for epoch in range(1, Params.epoch + 1):
            print(epoch, " .epoch başladı ...")
            # train
            _train_loss, _train_f1_micro, _train_f1_macro, _train_f1_weighted = Main.run_train(train_dataloader)
            train_loss.append(_train_loss)
            train_f1_micro.append(_train_f1_micro)
            train_f1_macro.append(_train_f1_macro)
            train_f1_weighted.append(_train_f1_weighted)

            # test
            _test_loss, _test_f1_micro, _test_f1_macro, _test_f1_weighted = Main.run_test(test_dataloader)
            test_loss.append(_test_loss)
            test_f1_micro.append(_test_f1_micro)
            test_f1_macro.append(_test_f1_macro)
            test_f1_weighted.append(_test_f1_weighted)

            # info
            print("train loss -> ", _train_loss)
            print("train f1-micro -> ", _train_f1_micro)
            print("train f1-macro -> ", _train_f1_macro)
            print("train f1-weighted -> ", _train_f1_weighted)

            print("test loss -> ", _test_loss)
            print("test f1-micro -> ", _test_f1_micro)
            print("test f1-macro -> ", _test_f1_macro)
            print("test f1-weighted -> ", _test_f1_weighted)

        # plot
        Main.plot_loss(train_loss, test_loss)
        Main.plot_f1(train_f1_micro, test_f1_micro, mode="micro")
        Main.plot_f1(train_f1_macro, test_f1_macro, mode="macro")
        Main.plot_f1(train_f1_weighted, test_f1_weighted, mode="weighted")

if __name__ == '__main__':

    print("cuda available: ", torch.cuda.is_available())

    data = Main.load_dataset()
    train_dataloader, test_dataloader = Main.get_dataloaders(data)
    Main.run_train_test(train_dataloader, test_dataloader)
