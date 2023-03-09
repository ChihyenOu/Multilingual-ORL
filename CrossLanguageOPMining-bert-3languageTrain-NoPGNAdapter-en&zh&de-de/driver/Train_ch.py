import sys
import logging
sys.path.extend(["../../","../","./"])
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from driver.Config import *
from driver.Model import *
from driver.Labeler import *
from data.Dataloader import *
import pickle
import os
import re
from driver.BertTokenHelper import BertTokenHelper
from driver.BertModel import BertExtractor

from driver.language_mlp import LanguageMLP

from driver.modeling import BertModel as AdapterBERTModel
from driver.modeling import BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from driver.adapterPGNBERT import AdapterPGNBertModel
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(level=logging.DEBUG, filename='myLog.log', filemode='w')

record_exact_dev_scores = []
record_exact_dev_recalls = []
record_exact_dev_precisions = []
loss_record = []

record_exact_test_scores = []
record_exact_test_recalls = []
record_exact_test_precisions = []

def train(data, dev_data, test_data, labeler, vocab, config, bert, language_embedder):
    optimizer = Optimizer(filter(lambda p: p.requires_grad, labeler.model.parameters()), config)
    optimizer_lang = Optimizer(filter(lambda p: p.requires_grad, language_embedder.parameters()), config)
    optimizer_bert = AdamW(filter(lambda p: p.requires_grad, bert.parameters()), lr=5e-6, eps=1e-8)
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0,
                                                     num_training_steps=config.train_epochs * batch_num)

    global_step = 0
    best_score = -1
    
    for epoch in range(config.train_epochs):
        total_stats = Statistics()
        print('Epoch: ' + str(epoch))
        batch_iter = 0

        for onebatch in data_iter(data, config.train_batch_size, True):
            words, extwords, predicts, inmasks, labels, outmasks, \
                bert_indices_tensor, bert_segments_tensor, bert_pieces_tensor, lang_ids = \
                        batch_data_variable(onebatch, vocab)

            language_embedder.train()
            bert.train()
            labeler.model.train()

            if config.use_cuda:
                bert_indices_tensor = bert_indices_tensor.cuda()
                bert_segments_tensor = bert_segments_tensor.cuda()
                bert_pieces_tensor = bert_pieces_tensor.cuda()

            lang_embedding = language_embedder(lang_ids)
            pgnbert_hidden = bert(input_ids=bert_indices_tensor, token_type_ids=bert_segments_tensor, 
                                        bert_pieces=bert_pieces_tensor, lang_embedding=lang_embedding)


            labeler.forward(words, extwords, predicts, inmasks, pgnbert_hidden)
            loss, stat = labeler.compute_loss(labels, outmasks)
            loss = loss / config.update_every
            print("loss: ", loss.item())
            loss_record.append(loss.item())
            loss.backward()

            total_stats.update(stat)
            total_stats.print_out(global_step, epoch, batch_iter, batch_num)
            batch_iter += 1
            
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, labeler.model.parameters()), \
                                        max_norm=config.clip)
                optimizer.step()
                optimizer_lang.step()
                optimizer_bert.step()
                labeler.model.zero_grad()
                optimizer_lang.zero_grad()
                optimizer_bert.zero_grad()
                global_step += 1
            
            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                logging.info("Epoch: %d, Batch_iter: %d" %(epoch, batch_iter))

                gold_num, predict_num, correct_num = \
                    evaluate(dev_data, labeler, vocab, config.dev_file + '.' + str(global_step))
                dev_score = 200.0 * correct_num / (gold_num + predict_num) if correct_num > 0 else 0.0
                exact_dev_recall = 100.0 * correct_num / gold_num if correct_num > 0 else 0.0
                exact_dev_precision = 100.0 * correct_num / predict_num if correct_num > 0 else 0.0
                logging.info("Dev: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (correct_num, gold_num, exact_dev_recall,  \
                       correct_num, predict_num, exact_dev_precision, \
                       dev_score))
                print("Dev: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (correct_num, gold_num, 100.0 * correct_num / gold_num if correct_num > 0 else 0.0,  \
                       correct_num, predict_num, 100.0 * correct_num / predict_num if correct_num > 0 else 0.0, \
                       dev_score))
                record_exact_dev_scores.append(dev_score)
                record_exact_dev_recalls.append(exact_dev_recall)
                record_exact_dev_precisions.append(exact_dev_precision)

                test_gold_num, test_predict_num, test_correct_num = \
                    evaluate(test_data, labeler, vocab, config.test_file + '.' + str(global_step))
                test_score = 200.0 * test_correct_num / (test_gold_num + test_predict_num) \
                                if test_correct_num > 0 else 0.0
                exact_test_recall = 100.0 * test_correct_num / test_gold_num if test_correct_num > 0 else 0.0
                exact_test_precision = 100.0 * test_correct_num / test_predict_num if test_correct_num > 0 else 0.0
                logging.info("Test: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (test_correct_num, test_gold_num,  \
                       exact_test_recall, \
                       test_correct_num, test_predict_num, \
                       exact_test_precision, \
                       test_score))
                print("Test: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (test_correct_num, test_gold_num,  \
                       100.0 * test_correct_num / test_gold_num if test_correct_num > 0 else 0.0, \
                       test_correct_num, test_predict_num, \
                       100.0 * test_correct_num / test_predict_num if test_correct_num > 0 else 0.0, \
                       test_score))
                record_exact_test_scores.append(test_score)
                record_exact_test_recalls.append(exact_test_recall)
                record_exact_test_precisions.append(exact_test_precision)

                if dev_score > best_score:
                    print("Exceed best score: history = %.2f, current = %.2f" %(best_score, dev_score))
                    best_score = dev_score
                    if config.save_after > 0 and epoch > config.save_after:
                        torch.save(labeler.model.state_dict(), config.save_model_path)

    logging.info("Loss scores: %s", loss_record)
    logging.info("record_exact_dev_scores: %s", record_exact_dev_scores)
    logging.info("record_exact_dev_recalls: %s", record_exact_dev_recalls)
    logging.info("record_exact_dev_precisions: %s", record_exact_dev_precisions)
    logging.info("record_exact_test_scores: %s", record_exact_test_scores)
    logging.info("record_exact_test_recalls: %s", record_exact_test_recalls)
    logging.info("record_exact_test_precisions: %s", record_exact_test_precisions)

def evaluate(data, labeler, vocab, outputFile):
    start = time.time()
    labeler.model.eval()
    language_embedder.eval()
    bert.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    total_gold_entity_num, total_predict_entity_num, total_correct_entity_num = 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False, False):
        words, extwords, predicts, inmasks, labels, outmasks, \
        bert_indices_tensor, bert_segments_tensor, bert_pieces_tensor, lang_ids  = \
            batch_data_variable(onebatch, vocab)
        if config.use_cuda:
            bert_indices_tensor = bert_indices_tensor.cuda()
            bert_segments_tensor = bert_segments_tensor.cuda()
            bert_pieces_tensor = bert_pieces_tensor.cuda()
        count = 0
        lang_embedding = language_embedder(lang_ids)
        bert_hidden = bert(input_ids=bert_indices_tensor, token_type_ids=bert_segments_tensor, bert_pieces=bert_pieces_tensor, lang_embedding=lang_embedding)
        predict_labels = labeler.label(words, extwords, predicts, inmasks, bert_hidden)
        for result in batch_variable_srl(onebatch, predict_labels, vocab):
            printSRL(output, result)
            gold_entity_num, predict_entity_num, correct_entity_num, \
            gold_agent_entity_num, predict_agent_entity_num, correct_agent_entity_num, \
            gold_target_entity_num, predict_target_entity_num, correct_target_entity_num = evalSRLExact(onebatch[count],
                                                                                                        result)
            total_gold_entity_num += gold_entity_num
            total_predict_entity_num += predict_entity_num
            total_correct_entity_num += correct_entity_num
            count += 1

    output.close()

    #R = np.float64(total_correct_entity_num) * 100.0 / np.float64(total_gold_entity_num)
    #P = np.float64(total_correct_entity_num) * 100.0 / np.float64(total_predict_entity_num)
    #F = np.float64(total_correct_entity_num) * 200.0 / np.float64(total_gold_entity_num + total_predict_entity_num)


    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))

    return total_gold_entity_num, total_predict_entity_num, total_correct_entity_num


class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        #self.optim = torch.optim.Adadelta(parameter, lr=1.0, rho=0.95)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()

if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = creat_vocab(config.train_file, config.min_occur_count)
    # vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)

    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    language_embedder = LanguageMLP(config=config)

    model = eval(config.model)(vocab, config) #, vec)

    bert_config = BertConfig.from_json_file(config.bert_config_path)
    bert_config.use_adapter = config.use_adapter
    bert_config.use_language_emb = config.use_language_emb
    bert_config.num_adapters = config.num_adapters
    bert_config.adapter_size = config.adapter_size
    bert_config.language_emb_size = config.language_emb_size
    bert_config.num_language_features = config.language_features
    bert_config.nl_project = config.nl_project
    # BERT
    bert = AdapterBERTModel.from_pretrained(config.bert_path, config=bert_config)

    
    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()
        bert = bert.cuda()
        language_embedder = language_embedder.cuda()

    labeler = SRLLabeler(model)

    bert_token = BertTokenHelper(config.bert_path)

    in_language_list = config.in_langs
    out_language_list = config.out_langs

    lang_dic = {}
    lang_dic['in'] = in_language_list
    lang_dic['oov'] = out_language_list

    data = read_corpus(config.train_file, bert_token, lang_dic)
    dev_data = read_corpus(config.dev_file, bert_token, lang_dic)
    test_data = read_corpus(config.test_file, bert_token, lang_dic)
    print("Finish code test!")

    train(data, dev_data, test_data, labeler, vocab, config, bert, language_embedder)
    