"""Action"""

import sys
import os, logging
import time
import gc
from functools import wraps

import torch
import torch.nn as nn
import sklearn
import sklearn.metrics
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from utils.linear_assignment_ import linear_assignment
import transformers
from torch.nn import functional as F
logging.getLogger().setLevel(logging.INFO)


class Action:
    """Action
    """
    def __init__(self, config: dict):
        super(Action, self).__init__()
        self.config = config
        self.optimizer = None
        self.tokenizer = None

    def plot_loss(self, loss, epoch, writer, is_train: bool):
        """Plot loss"""
        prefixed = "Train/" if is_train else "Val/"
        writer.add_scalar(prefixed + "loss", loss, epoch)

    def plot_metrics(self, metrics: dict, epoch: int, writer, is_train: bool):
        """Plot metrics"""
        tags = ["nmi", "ari", "acc"]
        prefixed = "Train/" if is_train else "Val/"
        for tag, metric in zip(tags, metrics):
            writer.add_scalar(prefixed + tag, metrics[metric], epoch)

    def get_lr_scheduler(self, optimizer, num_training_steps):
        self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                 num_warmup_steps=min(10000, max(100,int(num_training_steps / 10))),
                 num_training_steps=num_training_steps,)
        return self.lr_scheduler

    def set_tensorboard(self, path):
        self.tensorboard = SummaryWriter(log_dir=path)
    
    def get_tokenizer(self, path):
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(path)
        return self.tokenizer

    def get_pretrained_config(self, path):
        self.pretrained_config = transformers.RealmConfig.from_pretrained(path)
        return self.pretrained_config

    def get_optimizer(self, model, learning_rate: float):
        """Get optimizer
        :param model:
        :param learning_rate:
        :return: optimizer
        """
        #self.optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, \
                weight_decay=0.01, eps=1e-6,)
        return self.optimizer

    #marginal los loss
    def marginal_log_loss(self, logits, is_correct):
        '''
        :param logits:            old generator loss, [batch_size, top-k]
        :param is_correct:    generator_correct :- softmax(relevance_score), [batch_size, top-k]
        '''
        def mask_to_score(mask, dtype=torch.float32):
            print('mask', mask.shape, mask, mask.type(dtype))
            print('mask_to_score', (1.0 - mask.type(dtype)) * -10)
            return (1.0 - mask.type(dtype)) * -10
        log_numerator = torch.logsumexp(logits + mask_to_score(is_correct, dtype=logits.dtype), dim=-1)
        print('log_numerator', log_numerator.shape, log_numerator)

        log_denominator = torch.logsumexp(logits, dim=-1)
        print('log_denominator', log_denominator.shape, log_denominator)
        
        return log_denominator - log_numerator

    def get_net_loss_fn(self):
        #loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduce=False)
        loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction='none') # reduction : 'none' | 'mean' | 'sum'. 'none'
        #loss_func.to(self.config['device'])

        def cross_entropy_for_generator(logit, target):
            loss = loss_func(logit, target)
            return loss

        if 'generator' in self.config['net_type'] :
            self.net_loss_fn = cross_entropy_for_generator
        else: raise NotImplementedError('No loss function for {}'.format(self.config['model_type']))

    #entropy loss
    def get_retriver_loss_fn(self, logits):
        '''
        logits: relevance score
        [batch_size, searcher_beam_size]
        '''
        logit_log_softmax = F.log_softmax(logits, dim=1)
        logit_softmax = F.softmax(logits,dim=1)
        b = logit_softmax * logit_log_softmax
        b = -1.0 * b.sum(dim = -1)
        loss = torch.mean(b)

        #print('+++', loss, flush=True)

        return loss
        

    def get_loss_fn(self):
        """Get loss function, Cross entropy
        :return: loss function, mse
        """
        dataset = self.config["dataset"]
        if dataset == "jit_fake.json":
            retriever_loss = self.get_retriver_loss_fn
            self.get_net_loss_fn()
            net_loss = self.net_loss_fn
        elif dataset == "dstc.json":
            self.get_net_loss_fn()
            retriever_loss = self.net_loss_fn
            #retriever_loss = self.get_retriver_loss_fn
            #self.get_net_loss_fn()
            net_loss = self.net_loss_fn
        elif dataset == "GoogleNews_TS.txt":
            retriever_loss = self.get_retriver_loss_fn
            net_loss = self.get_net_loss_fn
        return retriever_loss, net_loss


    def get_generated_targets(self, model, data, threshold):
        """Get generated labels by threshold
        :param model: model
        :param data: (batch_size, channels, height, width)
        :return: data: (batch_size, num_clusters)
        """
        with torch.no_grad():
            model.eval()
            # (batch_size, num_clusters)
            features = model(data)
            #print("feature", features.shape)
            #print("feature", features)
            # (batch_size, batch_size)
            dist_matrix = self.get_cos_similarity_distance(features)
            #print("@@@",threshold, dist_matrix, flush=True)
            # (batch_size, batch_size)
            sim_matrix = self.get_cos_similarity_by_threshold(dist_matrix,
                                                              threshold)
            #print("###", threshold, sim_matrix, flush=True)
            #sys.exit()
            return sim_matrix


    def __save_model__(self, model, optimizer, epoch, metrics, last=False):
        """save model"""
        val_metric = metrics
        acc = val_metric["acc"]
        checkpoint = dict()
        checkpoint['config'] = self.config
        checkpoint['epoch'] = epoch
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint["val_metric"] = val_metric

        model_name = "Exp{}-Epoch{}-Acc{:>5.2f}".format(
            self.config['experiment_index'], epoch, acc*100.0)

        model_dir = os.path.join(self.config['model_dir'],
                                 self.config['experiment_index'])

        if not last:
            model_name = model_name + "-Best"
            for f in os.listdir(model_dir):
                if f.split('-')[-1] == 'Best':
                    best_model_before = os.path.join(model_dir, f)
                    os.remove(os.path.join(best_model_before))
                    print('>> Delete best model before: {}'.\
                            format(best_model_before), flush=True)
        else:
            model_name = model_name + "-Last"
        model_path = os.path.join(model_dir, model_name)
        torch.save(checkpoint, model_path)
        print('>> Save best model: {}'.format(model_name), flush=True)

    def save_model(self, model, path, tokenizer, optimizer, config=None, block_records=None, extra_info=None):
        logging.info('SAVE: {}.'.format(path))
        os.makedirs(path, exist_ok=True)

        # save block records
        if block_records is not None:
            np.save(os.path.join(path, "block_records.npy"), block_records)
        # save tokenizer
        tokenizer.save_pretrained(path)
        model.save_pretrained(path)
        #if config:
        #    with open(os.path.join(path,'training_args.json'), 'w') as f:
        #        json.dump(config, f, ensure_ascii=False, indent=4)
    


    def set_external_memory_in_model(self, tokenizer=None, model=None, projected_size=None, batch_size=None, blok_records=None, mode=None):
        '''
        mode : training 
        '''
        if not(bool(mode)):
        #학습
            documents = [doc.decode() for doc in blok_records]
            def chunks(lst, n):
                """Yield successive n-sized chunks from lst."""
                for i in range(0, len(lst), n):
                    yield lst[i:i + n]

            for bidx, each in enumerate(chunks(documents, 128)):#batch_size==>128
                inputs = tokenizer(each, padding=True, truncation=True, return_tensors="pt").to(self.config['device'])
                with torch.no_grad():
                    projected_score = model.embedder(**inputs, return_dict=True).projected_score
                    model.block_emb[bidx*128: bidx*128+projected_score.size(0)] = projected_score


    def get_metrics(self, y_true: list, y_pred: list, k: list):
        """Get metrics"""
        metrics = dict()
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        assert y_true.shape == y_pred.shape
        metrics["nmi"] = get_nmi(y_true, y_pred)
        metrics["ari"] = get_ari(y_true, y_pred)
        #metrics["acc"], ind = get_acc(y_true, y_pred)
        metrics["acc_k"] = get_acc_at_k(y_true, y_pred, k)
        print(">>> NMI:{:.4f}\tARI:{:.4f}".format(metrics["nmi"], metrics["ari"]), flush=True)
        for x in k:
            print("\tACC{}:{:.4f}".format(x, metrics["acc_k"][x]), flush=True)
        return metrics

    def get_acc_func(self, opt='token-accuracy'):
        opt = opt.lower()
        if opt == 'token-accuracy': return get_token_accuracy
        elif opt == 'accuracy': return get_acc
        elif opt == 'acc_k': return get_acc_at_k
        else: raise NotImplementedError('OPTION {} is not supported in set_acc_func.'.format(opt))

def time_this(func, *args, **kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        sys.stdout.flush()
        elapse = time.time() - start_time
        print(">> Function: {} costs {:.4f}s".format(func.__name__, elapse),
              flush=True)
        sys.stdout.flush()
        return ret
    return wrapper

def gcollect(func, *args, **kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        gc.collect()
        print(">> Function: {} has been garbage collected".\
            format(func.__name__), flush=True)
        sys.stdout.flush()
        return ret
    return wrapper

def str2bool(val):
    """Convert str to bool"""
    value = None
    if val == 'True':
        value = True
    elif val == 'False':
        value = False
    else:
        raise ValueError
    return value

def get_nmi(y_true, y_pred):
    """Get normalized_mutual_info_score
    """
    nmi = sklearn.metrics.normalized_mutual_info_score(\
        y_true, y_pred, average_method="arithmetic")
    return nmi

def get_ari(y_true, y_pred):
    """Get metrics.adjusted_rand_score
    """
    return sklearn.metrics.adjusted_rand_score(y_true, y_pred)

def get_acc(y_true: np.ndarray, y_pred: np.ndarray):
    """Get acc, maximum weight matching in bipartite graphs
    ref: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn
                 /utils/linear_assignment_.py
    """
    assert y_true.size == y_pred.size
    dim = max(y_pred.max(), y_true.max()) + 1
    #cost_maxtrix = np.zeros((dim, dim), dtype=np.int8)
    cost_maxtrix = np.zeros((dim, dim), dtype=torch.int8 if self.config['device']=='mps' else torch.int64)

    for x, y in zip(y_pred, y_true): 
        cost_maxtrix[x, y] += 1
        ind = linear_assignment(cost_maxtrix.max() - cost_maxtrix)

    acc = sum([cost_maxtrix[x, y] for x, y in ind]) * 1.0 / y_pred.size
    return acc, ind

def get_token_accuracy(target, logits, target_length=None):
    prediction = torch.argmax(logits, dim=-1)
    acc, count = 0, 0
    for pindex in range(prediction.shape[0]):
        gold = target[pindex,:target_length[pindex]]
        pred = prediction[pindex,:target_length[pindex]]
        if len(gold) > len(pred):
            pad = [0]*(len(gold)-len(pred))
            pad = torch.tensor(pad).to(prediction.device)
            pred = torch.cat((pred,pad),dim=0)
        elif len(pred) > len(gold):
            pad = [0]*(len(pred)-len(gold))
            pad = torch.tensor(pad).to(prediction.device)
            gold = torch.cat((gold,pad),dim=0)
        acc += int(sum(pred == gold))
        count += len(pred)
    return acc/count

def get_acc_at_k(y_true, y_pred, topk=[1]): 
    ''' 
    precision at k
    k is a list
    '''
    label_list = np.array([a for a in range(y_pred.shape[-1])])
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    metric = dict()
    for a in topk:
        acc_k = sklearn.metrics.top_k_accuracy_score(y_true, y_pred, labels=label_list, k=a)
        metric[a] = acc_k
    return metric

def getBack(var_grad_fn):
    print('===================== Tracing back tensors: =======================')
    print('\n\n', var_grad_fn, "::", var_grad_fn.next_functions)
    for i, n in enumerate(var_grad_fn.next_functions):
        print(i, '==>', n)
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor.size(), tensor.dtype, tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                print("+++++++ Attribute Error ")
                getBack(n[0])

