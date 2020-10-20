
import torch
import os

class Params(object):

    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    plot_dir = "plot"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    FULL_FINETUNING = True
    use_cuda = torch.cuda.is_available()
    dataset_dir = os.path.join(data_dir, "4900_news.xlsx")
    test_split_rate = 0.3
    bert_model_name = "dbmdz/bert-base-turkish-cased"

    tokenizer = None
    model = None
    optimizer = None
    scheduler = None
    label2id = {}
    epoch = 30
    batch_size = 32
    max_sentence_size = None
    batch_shuffle = True
    AdamW_lr = 3e-5
    AdamW_eps = 1e-8
    max_grad_norm = 1.0 # clip gradient
    weight_decay = 0.0 # 0.9 / 0.0: default

