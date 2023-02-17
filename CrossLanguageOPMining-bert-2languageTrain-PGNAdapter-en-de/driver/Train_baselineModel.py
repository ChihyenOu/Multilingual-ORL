# encoding:utf-8

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
record_exact_dev_agent_scores = []
record_exact_dev_agent_recalls = []
record_exact_dev_agent_precisions = []
record_exact_dev_target_scores = []
record_exact_dev_target_recalls = []
record_exact_dev_target_precisions = []

record_binary_dev_scores = []
record_binary_dev_recalls = []
record_binary_dev_precisions = []
record_binary_dev_agent_scores = []
record_binary_dev_agent_recalls = []
record_binary_dev_agent_precisions = []
record_binary_dev_target_scores = []
record_binary_dev_target_recalls = []
record_binary_dev_target_precisions = []

record_prop_dev_scores = []
record_prop_dev_recalls = []
record_prop_dev_precisions = []
record_prop_dev_agent_scores = []
record_prop_dev_agent_recalls = []
record_prop_dev_agent_precisions = []
record_prop_dev_target_scores = []
record_prop_dev_target_recalls = []
record_prop_dev_target_precisions = []


record_exact_test_scores = []
record_exact_test_recalls = []
record_exact_test_precisions = []
record_exact_test_agent_scores = []
record_exact_test_agent_recalls = []
record_exact_test_agent_precisions = []
record_exact_test_target_scores = []
record_exact_test_target_recalls = []
record_exact_test_target_precisions = []

record_binary_test_scores = []
record_binary_test_recalls = []
record_binary_test_precisions = []
record_binary_test_agent_scores = []
record_binary_test_agent_recalls = []
record_binary_test_agent_precisions = []
record_binary_test_target_scores = []
record_binary_test_target_recalls = []
record_binary_test_target_precisions = []

record_prop_test_scores = []
record_prop_test_recalls = []
record_prop_test_precisions = []
record_prop_test_agent_scores = []
record_prop_test_agent_recalls = []
record_prop_test_agent_precisions = []
record_prop_test_target_scores = []
record_prop_test_target_recalls = []
record_prop_test_target_precisions = []

loss_record = []

def train(data, dev_data, test_data, labeler, vocab, config, bert, language_embedder):
   # NEW VERSION
    optimizer_label = torch.optim.AdamW(filter(lambda p: p.requires_grad, labeler.model.parameters()), lr=config.learning_rate, \
                                                betas=(config.beta_1, config.beta_2), eps=config.epsilon)    
    decay, decay_step = config.decay, config.decay_steps
    l = lambda epoch: decay ** (epoch // decay_step)
    scheduler_label = torch.optim.lr_scheduler.LambdaLR(optimizer_label, lr_lambda=l)

    optimizer_lang = torch.optim.AdamW(filter(lambda p: p.requires_grad, language_embedder.parameters()), lr=config.learning_rate, \
                                                betas=(config.beta_1, config.beta_2), eps=config.epsilon)
    scheduler_lang = torch.optim.lr_scheduler.LambdaLR(optimizer_lang, lr_lambda=l)
    ####
    
    # change from AdamW to torch.optim.AdamW
    optimizer_bert = torch.optim.AdamW(filter(lambda p: p.requires_grad, bert.parameters()), lr=5e-6, eps=1e-8)
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    # scheduler_bert = WarmupLinearSchedule(optimizer_bert, warmup_steps=0, t_total=config.train_epochs * batch_num)
    scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0, num_training_steps=config.train_epochs * batch_num)


    global_step = 0
    best_score = -1
    # batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    for epoch in range(config.train_epochs): # iter -> epoch; config.train_iters -> config.train_epochs
        total_stats = Statistics()
        print('Epoch: ' + str(epoch))
        batch_iter = 0
        ii = 1

        language_embedder.train()
        bert.train()
        labeler.model.train()

        for onebatch in data_iter(data, config.train_batch_size, True):
                print("Iteration: ", ii)
                ii += 1
                
                words, extwords, predicts, inmasks, labels, outmasks, \
                bert_indices_tensor, bert_segments_tensor, bert_pieces_tensor, lang_ids = \
                        batch_data_variable(onebatch, vocab)
                
                
                if config.use_cuda:
                        bert_indices_tensor = bert_indices_tensor.cuda()
                        bert_segments_tensor = bert_segments_tensor.cuda()
                        bert_pieces_tensor = bert_pieces_tensor.cuda()

                # Forward pass
                lang_embedding = language_embedder(lang_ids)
                
                ## Baseline Model - BERTModel  & PGNAdaptor
                # bert_hidden = bert(input_ids=bert_indices_tensor, token_type_ids=bert_segments_tensor, 
                                        # bert_pieces=bert_pieces_tensor, lang_embedding=lang_embedding)
                # labeler.forward(words, extwords, predicts, inmasks, bert_hidden)
                
                # PGNAdaptor
                pgnbert_hidden = bert(input_ids=bert_indices_tensor, token_type_ids=bert_segments_tensor, 
                                        bert_pieces=bert_pieces_tensor, lang_embedding=lang_embedding)
                # print(pgnbert_hidden.size())
                labeler.forward(words, extwords, predicts, inmasks, pgnbert_hidden) 
                
                loss, stat = labeler.compute_loss(labels, outmasks)
                loss = loss / config.update_every
                print("loss: ", loss.item())
                loss_record.append(loss.item())
                
                optimizer_lang.zero_grad() ## ADD ZERO_GRAD
                optimizer_bert.zero_grad() ## ADD ZERO_GRAD
                optimizer_label.zero_grad() ## ADD ZERO_GRAD

                # print("optimizer_label params: ", optimizer_label.param_groups)

                # Backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, labeler.model.parameters()), \
                                                max_norm=config.clip)
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, bert.parameters()), \
                                                max_norm=config.clip)
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, language_embedder.parameters()), \
                                                max_norm=config.clip)
                                                
                total_stats.update(stat)
                total_stats.print_out(global_step, epoch, batch_iter, batch_num) # iter -> epoch

                # STEP 2
                optimizer_lang.step()
                optimizer_bert.step()
                optimizer_label.step() ## ADD step() to update parameters after each iteration
                batch_iter += 1
                
                if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                        # Update scheduler for learning rate at the end of each epoch AND after optimizer.step()
                        # print("optimizer_label params: ", optimizer_label.param_groups)
                        # Schduler step
                        scheduler_lang.step()
                        scheduler_bert.step()
                        scheduler_label.step()
                        global_step += 1

                
                if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                        gold_num, predict_num, correct_num, \
                        gold_agent_num, predict_agent_num, correct_agent_num, \
                        gold_target_num, predict_target_num, correct_target_num, \
                        binary_gold_num, binary_predict_num, binary_gold_correct_num, binary_predict_correct_num, \
                        binary_gold_agent_num, binary_predict_agent_num, binary_gold_correct_agent_num, binary_predict_correct_agent_num, \
                        binary_gold_target_num, binary_predict_target_num, binary_gold_correct_target_num, binary_predict_correct_target_num, \
                        prop_gold_num, prop_predict_num, prop_gold_correct_num, prop_predict_correct_num, \
                        prop_gold_agent_num, prop_predict_agent_num, prop_gold_correct_agent_num, prop_predict_correct_agent_num, \
                        prop_gold_target_num, prop_predict_target_num, prop_gold_correct_target_num, prop_predict_correct_target_num \
                        = evaluate(dev_data, labeler, vocab, config.target_dev_file + '.' + str(global_step))

                        print("Global step: ", global_step)

                        logging.info("Epoch: %d, Batch_iter: %d" %(epoch, batch_iter))
                
                        dev_score = 200.0 * correct_num / (gold_num + predict_num) if correct_num > 0 else 0.0
                        exact_dev_recall = 100.0 * correct_num / gold_num if correct_num > 0 else 0.0
                        exact_dev_precision = 100.0 * correct_num / predict_num if correct_num > 0 else 0.0
                        logging.info("Exact Dev: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                        (correct_num, gold_num, exact_dev_recall, \
                        correct_num, predict_num, exact_dev_precision, \
                        dev_score))
                        print("Exact Dev: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                        (correct_num, gold_num, exact_dev_recall, \
                        correct_num, predict_num, exact_dev_precision, \
                        dev_score))
                        record_exact_dev_scores.append(dev_score)
                        record_exact_dev_recalls.append(exact_dev_recall)
                        record_exact_dev_precisions.append(exact_dev_precision)

                        dev_agent_score = 200.0 * correct_agent_num / (
                                gold_agent_num + predict_agent_num) if correct_agent_num > 0 else 0.0
                        exact_dev_agent_recall = 100.0 * correct_agent_num / gold_agent_num if correct_agent_num > 0 else 0.0
                        exact_dev_agent_precision = 100.0 * correct_agent_num / predict_agent_num if correct_agent_num > 0 else 0.0
                        logging.info("Exact Dev Agent: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                        (correct_agent_num, gold_agent_num,
                        exact_dev_agent_recall, \
                        correct_agent_num, predict_agent_num,
                        exact_dev_agent_precision, \
                        dev_agent_score))
                        record_exact_dev_agent_scores.append(dev_agent_score)
                        record_exact_dev_agent_recalls.append(exact_dev_agent_recall)
                        record_exact_dev_agent_precisions.append(exact_dev_agent_precision)

                        dev_target_score = 200.0 * correct_target_num / (
                                gold_target_num + predict_target_num) if correct_target_num > 0 else 0.0
                        exact_dev_target_recall = 100.0 * correct_target_num / gold_target_num if correct_target_num > 0 else 0.0
                        exact_dev_target_precision = 100.0 * correct_target_num / predict_target_num if correct_target_num > 0 else 0.0
                        logging.info("Exact Dev Target: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                        (correct_target_num, gold_target_num,
                        exact_dev_target_recall, \
                        correct_target_num, predict_target_num,
                        exact_dev_target_precision, \
                        dev_target_score))
                        record_exact_dev_target_scores.append(dev_target_score)
                        record_exact_dev_target_recalls.append(exact_dev_target_recall)
                        record_exact_dev_target_precisions.append(exact_dev_target_precision)
                        #print()

                        binary_dev_P = binary_predict_correct_num / binary_predict_num if binary_predict_num > 0 else 0.0
                        binary_dev_R = binary_gold_correct_num / binary_gold_num if binary_gold_num > 0 else 0.0
                        dev_binary_score = 200 * binary_dev_P * binary_dev_R / (
                                binary_dev_P + binary_dev_R) if binary_dev_P + binary_dev_R > 0 else 0.0
                        logging.info("Binary Dev: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                        (binary_gold_correct_num, binary_gold_num, 100.0 * binary_dev_R, \
                        binary_predict_correct_num, binary_predict_num, 100.0 * binary_dev_P, \
                        dev_binary_score))
                        record_binary_dev_scores.append(dev_binary_score)
                        record_binary_dev_recalls.append(100.0 * binary_dev_R)
                        record_binary_dev_precisions.append(100.0 * binary_dev_P)

                        binary_dev_agent_P = binary_predict_correct_agent_num / binary_predict_agent_num if binary_predict_agent_num > 0 else 0.0
                        binary_dev_agent_R = binary_gold_correct_agent_num / binary_gold_agent_num if binary_gold_agent_num > 0 else 0.0
                        dev_binary_agent_score = 200 * binary_dev_agent_P * binary_dev_agent_R / (
                                binary_dev_agent_P + binary_dev_agent_R) if binary_dev_agent_P + binary_dev_agent_R > 0 else 0.0
                        logging.info("Binary Dev Agent: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                        (binary_gold_correct_agent_num, binary_gold_agent_num, 100.0 * binary_dev_agent_R, \
                        binary_predict_correct_agent_num, binary_predict_agent_num, 100.0 * binary_dev_agent_P, \
                        dev_binary_agent_score))
                        record_binary_dev_agent_scores.append(dev_binary_agent_score)
                        record_binary_dev_agent_recalls.append(100.0 * binary_dev_agent_R)
                        record_binary_dev_agent_precisions.append(100.0 * binary_dev_agent_P)

                        binary_dev_target_P = binary_predict_correct_target_num / binary_predict_target_num if binary_predict_target_num > 0 else 0.0
                        binary_dev_target_R = binary_gold_correct_target_num / binary_gold_target_num if binary_gold_target_num > 0 else 0.0
                        dev_binary_target_score = 200 * binary_dev_target_P * binary_dev_target_R / (
                                binary_dev_target_P + binary_dev_target_R) if binary_dev_target_P + binary_dev_target_R > 0 else 0.0
                        logging.info("Binary Dev Target: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                        (binary_gold_correct_target_num, binary_gold_target_num, 100.0 * binary_dev_target_R, \
                        binary_predict_correct_target_num, binary_predict_target_num, 100.0 * binary_dev_target_P, \
                        dev_binary_target_score))
                        record_binary_dev_target_scores.append(dev_binary_target_score)
                        record_binary_dev_target_recalls.append(100.0 * binary_dev_target_R)
                        record_binary_dev_target_precisions.append(100.0 * binary_dev_target_P)
                        #print()

                        prop_dev_P = prop_predict_correct_num / prop_predict_num if prop_predict_num > 0 else 0.0
                        prop_dev_R = prop_gold_correct_num / prop_gold_num if prop_gold_num > 0 else 0.0
                        dev_prop_score = 200 * prop_dev_P * prop_dev_R / (
                                prop_dev_P + prop_dev_R) if prop_dev_P + prop_dev_R > 0 else 0.0
                        logging.info("Prop Dev: Recall = %.2f/%d = %.2f, Precision = %.2f/%d =%.2f, F-measure = %.2f" % \
                        (prop_gold_correct_num, prop_gold_num, 100.0 * prop_dev_R, \
                        prop_predict_correct_num, prop_predict_num, 100.0 * prop_dev_P, \
                        dev_prop_score))
                        record_prop_dev_scores.append(dev_prop_score)
                        record_prop_dev_recalls.append(100.0 * prop_dev_R)
                        record_prop_dev_precisions.append(100.0 * prop_dev_P)

                        prop_dev_agent_P = prop_predict_correct_agent_num / prop_predict_agent_num if prop_predict_agent_num > 0 else 0.0
                        prop_dev_agent_R = prop_gold_correct_agent_num / prop_gold_agent_num if prop_gold_agent_num > 0 else 0.0
                        dev_prop_agent_score = 200 * prop_dev_agent_P * prop_dev_agent_R / (
                                prop_dev_agent_P + prop_dev_agent_R) if prop_dev_agent_P + prop_dev_agent_R > 0 else 0.0
                        logging.info("Prop Dev Agent: Recall = %.2f/%d = %.2f, Precision = %.2f/%d =%.2f, F-measure = %.2f" % \
                        (prop_gold_correct_agent_num, prop_gold_agent_num, 100.0 * prop_dev_agent_R, \
                        prop_predict_correct_agent_num, prop_predict_agent_num, 100.0 * prop_dev_agent_P, \
                        dev_prop_agent_score))
                        record_prop_dev_agent_scores.append(dev_prop_agent_score)
                        record_prop_dev_agent_recalls.append(100.0 * prop_dev_agent_R)
                        record_prop_dev_agent_precisions.append(100.0 * prop_dev_agent_P)

                        prop_dev_target_P = prop_predict_correct_target_num / prop_predict_target_num if prop_predict_target_num > 0 else 0.0
                        prop_dev_target_R = prop_gold_correct_target_num / prop_gold_target_num if prop_gold_target_num > 0 else 0.0
                        dev_prop_target_score = 200 * prop_dev_target_P * prop_dev_target_R / (
                                prop_dev_target_P + prop_dev_target_R) if prop_dev_target_P + prop_dev_target_R > 0 else 0.0
                        logging.info("Prop Dev Target: Recall = %.2f/%d = %.2f, Precision = %.2f/%d =%.2f, F-measure = %.2f" % \
                        (prop_gold_correct_target_num, prop_gold_target_num, 100.0 * prop_dev_target_R, \
                        prop_predict_correct_target_num, prop_predict_target_num, 100.0 * prop_dev_target_P, \
                        dev_prop_target_score))
                        record_prop_dev_target_scores.append(dev_prop_target_score)
                        record_prop_dev_target_recalls.append(100.0 * prop_dev_target_R)
                        record_prop_dev_target_precisions.append(100.0 * prop_dev_target_P)
                        # print()

                        
                        if dev_score > best_score:
                                print("Exceed best score: history = %.2f, current = %.2f" %(best_score, dev_score))
                                logging.info("Find better model in Epoch: %d, Batch_iter: %d" %(epoch, batch_iter))
                                logging.info("Exceed best score: history = %.2f, current = %.2f" %(best_score, dev_score))
                                
                                best_score = dev_score
                                if config.save_after > 0: # and epoch > config.save_after: # iter -> epoch
                                        logging.info("Test the model in Epoch: %d, Batch_iter: %d" %(epoch, batch_iter))
                                        torch.save(labeler.model.state_dict(), config.save_model_path)
                                        TestDataForBestModel(test_data, labeler, vocab, config, global_step)
                            
    logging.info("Loss scores: %s", loss_record)
    logging.info("record_exact_dev_scores: %s", record_exact_dev_scores)
    logging.info("record_exact_dev_recalls: %s", record_exact_dev_recalls)
    logging.info("record_exact_dev_precisions: %s", record_exact_dev_precisions)
    logging.info("record_exact_dev_agent_scores: %s", record_exact_dev_agent_scores)
    logging.info("record_exact_dev_agent_recalls: %s", record_exact_dev_agent_recalls)
    logging.info("record_exact_dev_agent_precisions: %s", record_exact_dev_agent_precisions)
    logging.info("record_exact_dev_target_scores: %s", record_exact_dev_target_scores)
    logging.info("record_exact_dev_target_recalls: %s", record_exact_dev_target_recalls)
    logging.info("record_exact_dev_target_precisions: %s", record_exact_dev_target_precisions)
    logging.info("record_binary_dev_scores: %s", record_binary_dev_scores)
    logging.info("record_binary_dev_recalls: %s", record_binary_dev_recalls)
    logging.info("record_binary_dev_precisions: %s", record_binary_dev_precisions)
    logging.info("record_binary_dev_agent_scores: %s", record_binary_dev_agent_scores)
    logging.info("record_binary_dev_agent_recalls: %s", record_binary_dev_agent_recalls)
    logging.info("record_binary_dev_agent_precisions: %s", record_binary_dev_agent_precisions)
    logging.info("record_binary_dev_target_scores: %s", record_binary_dev_target_scores)
    logging.info("record_binary_dev_target_recalls: %s", record_binary_dev_target_recalls)
    logging.info("record_binary_dev_target_precisions: %s", record_binary_dev_target_precisions)
    logging.info("record_prop_dev_scores: %s", record_prop_dev_scores)
    logging.info("record_prop_dev_recalls: %s", record_prop_dev_recalls)
    logging.info("record_prop_dev_precisions: %s", record_prop_dev_precisions)
    logging.info("record_prop_dev_agent_scores: %s", record_prop_dev_agent_scores)
    logging.info("record_prop_dev_agent_recalls: %s", record_prop_dev_agent_recalls)
    logging.info("record_prop_dev_agent_precisions: %s", record_prop_dev_agent_precisions)
    logging.info("record_prop_dev_target_scores: %s", record_prop_dev_target_scores)
    logging.info("record_prop_dev_target_recalls: %s", record_prop_dev_target_recalls)
    logging.info("record_prop_dev_target_precisions: %s", record_prop_dev_target_precisions)

    logging.info("record_exact_test_scores: %s", record_exact_test_scores)
    logging.info("record_exact_test_recalls: %s", record_exact_test_recalls)
    logging.info("record_exact_test_precisions: %s", record_exact_test_precisions)
    logging.info("record_exact_test_agent_scores: %s", record_exact_test_agent_scores)
    logging.info("record_exact_test_agent_recalls: %s", record_exact_test_agent_recalls)
    logging.info("record_exact_test_agent_precisions: %s", record_exact_test_agent_precisions)
    logging.info("record_exact_test_target_scores: %s", record_exact_test_target_scores)
    logging.info("record_exact_test_target_recalls: %s", record_exact_test_target_recalls)
    logging.info("record_exact_test_target_precisions: %s", record_exact_test_target_precisions)
    logging.info("record_binary_test_scores: %s", record_binary_test_scores)
    logging.info("record_binary_test_recalls: %s", record_binary_test_recalls)
    logging.info("record_binary_test_precisions: %s", record_binary_test_precisions)
    logging.info("record_binary_test_agent_scores: %s", record_binary_test_agent_scores)
    logging.info("record_binary_test_agent_recalls: %s", record_binary_test_agent_recalls)
    logging.info("record_binary_test_agent_precisions: %s", record_binary_test_agent_precisions)
    logging.info("record_binary_test_target_scores: %s", record_binary_test_target_scores)
    logging.info("record_binary_test_target_recalls: %s", record_binary_test_target_recalls)
    logging.info("record_binary_test_target_precisions: %s", record_binary_test_target_precisions)
    logging.info("record_prop_test_scores: %s", record_prop_test_scores)
    logging.info("record_prop_test_recalls: %s", record_prop_test_recalls)
    logging.info("record_prop_test_precisions: %s", record_prop_test_precisions)
    logging.info("record_prop_test_agent_scores: %s", record_prop_test_agent_scores)
    logging.info("record_prop_test_agent_recalls: %s", record_prop_test_agent_recalls)
    logging.info("record_prop_test_agent_precisions: %s", record_prop_test_agent_precisions)
    logging.info("record_prop_test_target_scores: %s", record_prop_test_target_scores)
    logging.info("record_prop_test_target_recalls: %s", record_prop_test_target_recalls)
    logging.info("record_prop_test_target_precisions: %s", record_prop_test_target_precisions)


def TestDataForBestModel(test_data, labeler, vocab, config, global_step):
    '''
        Test
    '''
    logging.info("Use current best model to test data!")
    test_gold_num, test_predict_num, test_correct_num, \
    test_gold_agent_num, test_predict_agent_num, test_correct_agent_num, \
    test_gold_target_num, test_predict_target_num, test_correct_target_num, \
    test_binary_gold_num, test_binary_predict_num, test_binary_gold_correct_num, test_binary_predict_correct_num, \
    test_binary_gold_agent_num, test_binary_predict_agent_num, test_binary_gold_correct_agent_num, test_binary_predict_correct_agent_num, \
    test_binary_gold_target_num, test_binary_predict_target_num, test_binary_gold_correct_target_num, test_binary_predict_correct_target_num, \
    test_prop_gold_num, test_prop_predict_num, test_prop_gold_correct_num, test_prop_predict_correct_num, \
    test_prop_gold_agent_num, test_prop_predict_agent_num, test_prop_gold_correct_agent_num, test_prop_predict_correct_agent_num, \
    test_prop_gold_target_num, test_prop_predict_target_num, test_prop_gold_correct_target_num, test_prop_predict_correct_target_num \
        = evaluate(test_data, labeler, vocab, config.target_test_file + '.' + str(global_step))

    test_score = 200.0 * test_correct_num / (test_gold_num + test_predict_num) \
        if test_correct_num > 0 else 0.0
    exact_test_recall = 100.0 * test_correct_num / test_gold_num if test_correct_num > 0 else 0.0
    exact_test_precision = 100.0 * test_correct_num / test_predict_num if test_correct_num > 0 else 0.0
    logging.info("Exact Test: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
            (test_correct_num, test_gold_num, \
            exact_test_recall, \
            test_correct_num, test_predict_num, \
            exact_test_precision, \
            test_score))
    record_exact_test_scores.append(test_score)
    record_exact_test_recalls.append(exact_test_recall)
    record_exact_test_precisions.append(exact_test_precision)


    test_agent_score = 200.0 * test_correct_agent_num / (
            test_gold_agent_num + test_predict_agent_num) if test_correct_agent_num > 0 else 0.0
    exact_test_agent_recall = 100.0 * test_correct_agent_num / test_gold_agent_num if test_correct_agent_num > 0 else 0.0
    exact_test_agent_precision = 100.0 * test_correct_agent_num / test_predict_agent_num if test_correct_agent_num > 0 else 0.0
    logging.info("Exact Test Agent: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
            (test_correct_agent_num, test_gold_agent_num,
            exact_test_agent_recall, \
            test_correct_agent_num, test_predict_agent_num,
            exact_test_agent_precision, \
            test_agent_score))
    record_exact_test_agent_scores.append(test_agent_score)
    record_exact_test_agent_recalls.append(exact_test_agent_recall)
    record_exact_test_agent_precisions.append(exact_test_agent_precision)

    test_target_score = 200.0 * test_correct_target_num / (
            test_gold_target_num + test_predict_target_num) if test_correct_target_num > 0 else 0.0
    exact_test_target_recall = 100.0 * test_correct_target_num / test_gold_target_num if test_correct_target_num > 0 else 0.0
    exact_test_target_precision = 100.0 * test_correct_target_num / test_predict_target_num if test_correct_target_num > 0 else 0.0
    logging.info("Exact Test Target: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
            (test_correct_target_num, test_gold_target_num,
            exact_test_target_recall, \
            test_correct_target_num, test_predict_target_num,
            exact_test_target_precision, \
            test_target_score))
    record_exact_test_target_scores.append(test_target_score)
    record_exact_test_target_recalls.append(exact_test_target_recall)
    record_exact_test_target_precisions.append(exact_test_target_precision)
    # print()

    binary_test_P = test_binary_predict_correct_num / test_binary_predict_num if test_binary_predict_num > 0 else 0.0
    binary_test_R = test_binary_gold_correct_num / test_binary_gold_num if test_binary_gold_num > 0 else 0.0
    binary_test_score = 200 * binary_test_P * binary_test_R / (
            binary_test_P + binary_test_R) if binary_test_P + binary_test_R > 0 else 0.0
    logging.info("Binary Test: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
            (test_binary_gold_correct_num, test_binary_gold_num, 100.0 * binary_test_R, \
            test_binary_predict_correct_num, test_binary_predict_num, 100.0 * binary_test_P, \
            binary_test_score))
    record_binary_test_scores.append(binary_test_score)
    record_binary_test_recalls.append(100.0 * binary_test_R)
    record_binary_test_precisions.append(100.0 * binary_test_P)

    binary_test_agent_P = test_binary_predict_correct_agent_num / test_binary_predict_agent_num if test_binary_predict_agent_num > 0 else 0.0
    binary_test_agent_R = test_binary_gold_correct_agent_num / test_binary_gold_agent_num if test_binary_gold_agent_num > 0 else 0.0
    binary_test_agent_score = 200 * binary_test_agent_P * binary_test_agent_R / (
            binary_test_agent_P + binary_test_agent_R) if binary_test_agent_P + binary_test_agent_R > 0 else 0.0
    logging.info("Binary Test Agent: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
            (test_binary_gold_correct_agent_num, test_binary_gold_agent_num, 100.0 * binary_test_agent_R, \
            test_binary_predict_correct_agent_num, test_binary_predict_agent_num,
            100.0 * binary_test_agent_P, \
            binary_test_agent_score))
    record_binary_test_agent_scores.append(binary_test_agent_score)
    record_binary_test_agent_recalls.append(100.0 * binary_test_agent_R)
    record_binary_test_agent_precisions.append(100.0 * binary_test_agent_P)

    binary_test_target_P = test_binary_predict_correct_target_num / test_binary_predict_target_num if test_binary_predict_target_num > 0 else 0.0
    binary_test_target_R = test_binary_gold_correct_target_num / test_binary_gold_target_num if test_binary_gold_target_num > 0 else 0.0
    binary_test_target_score = 200 * binary_test_target_P * binary_test_target_R / (
            binary_test_target_P + binary_test_target_R) if binary_test_target_P + binary_test_target_R > 0 else 0.0
    logging.info("Binary Test Target: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
            (test_binary_gold_correct_target_num, test_binary_gold_target_num, 100.0 * binary_test_target_R, \
            test_binary_predict_correct_target_num, test_binary_predict_target_num,
            100.0 * binary_test_target_P, \
            binary_test_target_score))
    record_binary_test_target_scores.append(binary_test_target_score)
    record_binary_test_target_recalls.append(100.0 * binary_test_target_R)
    record_binary_test_target_precisions.append(100.0 * binary_test_target_P)
    # print()

    prop_test_P = test_prop_predict_correct_num / test_prop_predict_num if test_prop_predict_num > 0 else 0.0
    prop_test_R = test_prop_gold_correct_num / test_prop_gold_num if test_prop_gold_num > 0 else 0.0
    prop_test_score = 200 * prop_test_P * prop_test_R / (
            prop_test_P + prop_test_R) if prop_test_P + prop_test_R > 0 else 0.0
    logging.info("Prop Test: Recall = %.2f/%d = %.2f, Precision = %.2f/%d =%.2f, F-measure = %.2f" % \
            (test_prop_gold_correct_num, test_prop_gold_num, 100.0 * prop_test_R, \
            test_prop_predict_correct_num, test_prop_predict_num, 100.0 * prop_test_P, \
            prop_test_score))
    record_prop_test_scores.append(prop_test_score)
    record_prop_test_recalls.append(100.0 * prop_test_R)
    record_prop_test_precisions.append(100.0 * prop_test_P)
    
    prop_test_agent_P = test_prop_predict_correct_agent_num / test_prop_predict_agent_num if test_prop_predict_agent_num > 0 else 0.0
    prop_test_agent_R = test_prop_gold_correct_agent_num / test_prop_gold_agent_num if test_prop_gold_agent_num > 0 else 0.0
    prop_test_agent_score = 200 * prop_test_agent_P * prop_test_agent_R / (
            prop_test_agent_P + prop_test_agent_R) if prop_test_agent_P + prop_test_agent_R > 0 else 0.0
    logging.info("prop Test Agent: Recall = %.2f/%d = %.2f, Precision = %.2f/%d =%.2f, F-measure = %.2f" % \
            (test_prop_gold_correct_agent_num, test_prop_gold_agent_num, 100.0 * prop_test_agent_R, \
            test_prop_predict_correct_agent_num, test_prop_predict_agent_num,
            100.0 * prop_test_agent_P, \
            prop_test_agent_score))
    record_prop_test_agent_scores.append(prop_test_agent_score)
    record_prop_test_agent_recalls.append(100.0 * prop_test_agent_R)
    record_prop_test_agent_precisions.append(100.0 * prop_test_agent_P)
    
    prop_test_target_P = test_prop_predict_correct_target_num / test_prop_predict_target_num if test_prop_predict_target_num > 0 else 0.0
    prop_test_target_R = test_prop_gold_correct_target_num / test_prop_gold_target_num if test_prop_gold_target_num > 0 else 0.0
    prop_test_target_score = 200 * prop_test_target_P * prop_test_target_R / (
            prop_test_target_P + prop_test_target_R) if prop_test_target_P + prop_test_target_R > 0 else 0.0
    logging.info("Prop Test Target: Recall = %.2f/%d = %.2f, Precision = %.2f/%d =%.2f, F-measure = %.2f" % \
            (test_prop_gold_correct_target_num, test_prop_gold_target_num,
            100.0 * prop_test_target_R, \
            test_prop_predict_correct_target_num, test_prop_predict_target_num,
            100.0 * prop_test_target_P, \
            prop_test_target_score))
    record_prop_test_target_scores.append(prop_test_target_score)
    record_prop_test_target_recalls.append(100.0 * prop_test_target_R)
    record_prop_test_target_precisions.append(100.0 * prop_test_target_P)


def evaluate(data, labeler, vocab, outputFile):
    start = time.time()
    labeler.model.eval()
    language_embedder.eval()
    bert.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    total_gold_entity_num, total_predict_entity_num, total_correct_entity_num = 0, 0, 0
    total_gold_agent_entity_num, total_predict_agent_entity_num, total_correct_agent_entity_num = 0, 0, 0
    total_gold_target_entity_num, total_predict_target_entity_num, total_correct_target_entity_num = 0, 0, 0

    binary_total_gold_entity_num, binary_total_predict_entity_num, binary_gold_total_correct_entity_num, binary_predict_total_correct_entity_num = 0, 0, 0, 0
    binary_total_gold_agent_entity_num, binary_total_predict_agent_entity_num, binary_gold_total_correct_agent_entity_num, binary_predict_total_correct_agent_entity_num = 0, 0, 0, 0
    binary_total_gold_target_entity_num, binary_total_predict_target_entity_num, binary_gold_total_correct_target_entity_num, binary_predict_total_correct_target_entity_num = 0, 0, 0, 0

    prop_total_gold_entity_num, prop_total_predict_entity_num, prop_gold_total_correct_entity_num, prop_predict_total_correct_entity_num = 0, 0, 0, 0
    prop_total_gold_agent_entity_num, prop_total_predict_agent_entity_num, prop_gold_total_correct_agent_entity_num, prop_predict_total_correct_agent_entity_num = 0, 0, 0, 0
    prop_total_gold_target_entity_num, prop_total_predict_target_entity_num, prop_gold_total_correct_target_entity_num, prop_predict_total_correct_target_entity_num = 0, 0, 0, 0

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
        # print("predicted labels: ", predict_labels)
        for result in batch_variable_srl(onebatch, predict_labels, vocab):
            printSRL(output, result)
            gold_entity_num, predict_entity_num, correct_entity_num, \
            gold_agent_entity_num, predict_agent_entity_num, correct_agent_entity_num, \
            gold_target_entity_num, predict_target_entity_num, correct_target_entity_num = evalSRLExact(onebatch[count],
                                                                                                        result)

            total_gold_entity_num += gold_entity_num
            total_predict_entity_num += predict_entity_num
            total_correct_entity_num += correct_entity_num

            total_gold_agent_entity_num += gold_agent_entity_num
            total_predict_agent_entity_num += predict_agent_entity_num
            total_correct_agent_entity_num += correct_agent_entity_num

            total_gold_target_entity_num += gold_target_entity_num
            total_predict_target_entity_num += predict_target_entity_num
            total_correct_target_entity_num += correct_target_entity_num

            binary_gold_entity_num, binary_predict_entity_num, binary_gold_correct_entity_num, binary_predict_correct_entity_num, \
            binary_gold_agent_entity_num, binary_predict_agent_entity_num, binary_gold_correct_agent_entity_num, binary_predict_correct_agent_entity_num, \
            binary_gold_target_entity_num, binary_predict_target_entity_num, binary_gold_correct_target_entity_num, binary_predict_correct_target_entity_num = evalSRLBinary(
                onebatch[count], result)

            binary_total_gold_entity_num += binary_gold_entity_num
            binary_total_predict_entity_num += binary_predict_entity_num
            binary_gold_total_correct_entity_num += binary_gold_correct_entity_num
            binary_predict_total_correct_entity_num += binary_predict_correct_entity_num

            binary_total_gold_agent_entity_num += binary_gold_agent_entity_num
            binary_total_predict_agent_entity_num += binary_predict_agent_entity_num
            binary_gold_total_correct_agent_entity_num += binary_gold_correct_agent_entity_num
            binary_predict_total_correct_agent_entity_num += binary_predict_correct_agent_entity_num

            binary_total_gold_target_entity_num += binary_gold_target_entity_num
            binary_total_predict_target_entity_num += binary_predict_target_entity_num
            binary_gold_total_correct_target_entity_num += binary_gold_correct_target_entity_num
            binary_predict_total_correct_target_entity_num += binary_predict_correct_target_entity_num

            prop_gold_entity_num, prop_predict_entity_num, prop_gold_correct_entity_num, prop_predict_correct_entity_num, \
            prop_gold_agent_entity_num, prop_predict_agent_entity_num, prop_gold_correct_agent_entity_num, prop_predict_correct_agent_entity_num, \
            prop_gold_target_entity_num, prop_predict_target_entity_num, prop_gold_correct_target_entity_num, prop_predict_correct_target_entity_num = evalSRLProportional(
                onebatch[count], result)

            prop_total_gold_entity_num += prop_gold_entity_num
            prop_total_predict_entity_num += prop_predict_entity_num
            prop_gold_total_correct_entity_num += prop_gold_correct_entity_num
            prop_predict_total_correct_entity_num += prop_predict_correct_entity_num

            prop_total_gold_agent_entity_num += prop_gold_agent_entity_num
            prop_total_predict_agent_entity_num += prop_predict_agent_entity_num
            prop_gold_total_correct_agent_entity_num += prop_gold_correct_agent_entity_num
            prop_predict_total_correct_agent_entity_num += prop_predict_correct_agent_entity_num

            prop_total_gold_target_entity_num += prop_gold_target_entity_num
            prop_total_predict_target_entity_num += prop_predict_target_entity_num
            prop_gold_total_correct_target_entity_num += prop_gold_correct_target_entity_num
            prop_predict_total_correct_target_entity_num += prop_predict_correct_target_entity_num
            count += 1

    output.close()

    #R = np.float64(total_correct_entity_num) * 100.0 / np.float64(total_gold_entity_num)
    #P = np.float64(total_correct_entity_num) * 100.0 / np.float64(total_predict_entity_num)
    #F = np.float64(total_correct_entity_num) * 200.0 / np.float64(total_gold_entity_num + total_predict_entity_num)


    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))
    logging.info("sentence num: %d,  parser time = %.2f " % (len(data), during_time))

    return total_gold_entity_num, total_predict_entity_num, total_correct_entity_num, \
           total_gold_agent_entity_num, total_predict_agent_entity_num, total_correct_agent_entity_num, \
           total_gold_target_entity_num, total_predict_target_entity_num, total_correct_target_entity_num, \
           binary_total_gold_entity_num, binary_total_predict_entity_num, binary_gold_total_correct_entity_num, binary_predict_total_correct_entity_num, \
           binary_total_gold_agent_entity_num, binary_total_predict_agent_entity_num, binary_gold_total_correct_agent_entity_num, binary_predict_total_correct_agent_entity_num, \
           binary_total_gold_target_entity_num, binary_total_predict_target_entity_num, binary_gold_total_correct_target_entity_num, binary_predict_total_correct_target_entity_num, \
           prop_total_gold_entity_num, prop_total_predict_entity_num, prop_gold_total_correct_entity_num, prop_predict_total_correct_entity_num, \
           prop_total_gold_agent_entity_num, prop_total_predict_agent_entity_num, prop_gold_total_correct_agent_entity_num, prop_predict_total_correct_agent_entity_num, \
           prop_total_gold_target_entity_num, prop_total_predict_target_entity_num, prop_gold_total_correct_target_entity_num, prop_predict_total_correct_target_entity_num

"""
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
        return self.scheduler.get_lr()"""


if __name__ == '__main__':
    random.seed(1000)  # 42, 666, 1000
    np.random.seed(1000)  # 42, 666, 1000
    torch.cuda.manual_seed(1000)  # 42, 666, 1000
    torch.manual_seed(1000)  # 42, 666, 1000

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='expdata/opinion.cfg')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=False)

    args, extra_args = argparser.parse_known_args()
    # print(args) # Namespace(config_file='expdata/opinion.cfg', thread=1, use_cuda=False)
    # print(extra_args) # []

    # if there is any extra_args, add them to config_file 
    # In our case, it doesn't so the config_file keeps the same
    config = Configurable(args.config_file, extra_args)

    # ADD one more language source file
    vocab = creat_vocab(config.source_train_file, config.target_train_file, config.min_occur_count) #modify
    # Remove the below line # pretrained_embeddings_file is not needed
    # vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file) 
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    # remove below two lines because it repeats twice?
    # args, extra_args = argparser.parse_known_args()
    # config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    
    language_embedder = LanguageMLP(config=config)

    # eval(expression): the content of expression is evaluated as a Python expression 
    # print(eval(config.model)) # <class 'driver.Model.BiLSTMCRFModel'>
    # # this eval() is not the model.eval() in PyTorch
    model = eval(config.model)(vocab, config) # Remove , vec
    # print(model) # BiLSTMCRFModel
    
    # bert = BertExtractor(config)
    bert_config = BertConfig.from_json_file(config.bert_config_path)
    bert_config.use_adapter = config.use_adapter
    bert_config.use_language_emb = config.use_language_emb
    bert_config.num_adapters = config.num_adapters
    bert_config.adapter_size = config.adapter_size
    bert_config.language_emb_size = config.language_emb_size
    bert_config.num_language_features = config.language_features
    bert_config.nl_project = config.nl_project
    # BERT
    bert = AdapterBERTModel.from_pretrained(config.bert_path, config=bert_config) # AdapterPGNBertModel xxxx

    # PGNBERT
    # bert = AdapterPGNBertModel(config.bert_path)
    # bert = AdapterPGNBertModel('bert-base-multilingual-cased', config=bert_config) # Use this version
    
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

    source_data = read_corpus(config.source_train_file, bert_token, lang_dic) #ADD
    target_data = read_corpus(config.target_train_file, bert_token, lang_dic) #ADD
    data = source_data + target_data #ADD
    dev_data = read_corpus(config.target_dev_file, bert_token, lang_dic) # Modify
    test_data = read_corpus(config.target_test_file, bert_token, lang_dic) #Modify
    print("Finish code test!")
    # PGNBERT
    train(data, dev_data, test_data, labeler, vocab, config, bert, language_embedder)
    
