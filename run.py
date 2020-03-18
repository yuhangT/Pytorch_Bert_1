import os
import shutil
import config
import time
import torch
from data_process import data_generator
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm, trange
from optimization import BERTAdam
import random
import numpy as np
from modeling import Discourage, DiscourageMask
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


####
# Name: print_model_result
# Function: 输出模型结果
####
def print_model_result(result, data_type='train'):
    # print("***** Eval results in " + data_type + "*****")
    # tmp = "\t"
    for key in sorted(result.keys()):
        # tmp += "%s = %s\t" % (key, str(result[key]).strip())
        print(" \t %s = %-5.5f" % (key, float(result[key])), end="")
    # print(tmp, end=" ")


####
# Name: model_eval
# Function: 在验证集和测试集上，评估模型
# return: 模型评估结果
####
def model_eval(model, data_loader, data_type='dev'):
    result_sum = {}
    nm_batch = 0
    labels_pred = np.array([])
    labels_true = np.array([])
    for step, batch in enumerate(tqdm(data_loader)):
        batch = tuple(t.to(config.device) for t in batch)
        model.eval()
        with torch.no_grad():
            _, pred = model(batch)
        pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
        labels_pred = np.append(labels_pred, pred)
        true = model.get_labels_data().detach().cpu().numpy()
        labels_true = np.append(labels_true, true)

        result_temp = model.get_result()
        result_sum['loss'] = result_sum.get('loss', 0) + result_temp['loss']
        nm_batch += 1

    result_sum["accuracy"] = accuracy_score(labels_true, labels_pred)
    result_sum["f1"] = f1_score(labels_true, labels_pred, average='macro')
    result_sum["loss"] = result_sum["loss"] / nm_batch
    with open(os.path.join(config.output_dir, config.MODEL_NAME + '_' + data_type + '_result.txt'), 'a+',
              encoding='utf-8') as writer:
        print("***** Eval results in " + data_type + "*****")
        for key in sorted(result_sum.keys()):
            print("%s = %s" % (key, str(result_sum[key])))
            writer.write("%s = %s\n" % (key, str(result_sum[key])))
        writer.write('\n')
    return result_sum


####
# Name: save_best_model
# Function: 在验证集或者训练集上, 保存loss最小或者准确度最高的模型参数。
####
def save_best_model(model, v, data_type='dev', use_accuracy=False):
    # 保存模型
    if not use_accuracy and data_type == 'dev':
        if config.eval_best_loss > v:
            config.eval_best_loss = v
            state = {'net': model.state_dict()}
            save_path = os.path.join(config.output_dir, config.MODEL_NAME + '_state_dict_' +
                                 data_type + '_loss_' + str(v) + '.model')
            print("Save.......")
            torch.save(state, save_path)
            config.train_best_loss_model = save_path

    # 以精确度作为评估标准
    if use_accuracy and data_type == 'dev':
        if config.eval_best_accuracy < v:
            config.eval_best_accuracy = v
            state = {'net': model.state_dict()}
            save_path = os.path.join(config.output_dir, config.MODEL_NAME + '_state_dict_'
                                     + data_type + '_ac_' + str(v) + '.model')
            print("Save.......")
            torch.save(state, save_path)
            config.train_best_accuracy_model = save_path


####
# Name: train
# Function: 训练并评估函数
####
def train(model):
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(config.device, n_gpu))
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model_it_self = model.module if hasattr(model, 'module') else model
    global_step = 0
    num_train_steps = data_generator.get_num_train_steps()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
            ]
    optimizer = BERTAdam(optimizer_grouped_parameters,
                             lr=config.learning_rate,
                                warmup=config.warmup_proportion,
                             t_total=num_train_steps)
    dev_loader = data_generator.get_dev_loader()
    train_loader = data_generator.get_train_loader()
    for epoch in trange(int(config.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            batch = tuple(t.to(config.device) for t in batch)
            loss, output = model(batch, global_step, -1)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            # opt.zero_grad()
            loss.backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1
            #if global_step % config.print_interval == 0:
            #    print_model_result(model_it_self.get_result())

            if global_step % config.eval_interval == 0 or global_step == num_train_steps:
                if config.do_eval:
                    print("\nepoch:{} global:{}\t".format(epoch, global_step))
                    eval_result = model_eval(model_it_self, dev_loader, data_type='dev')
                    # 保存模型，使用loss为评估标准
                    save_best_model(model_it_self, eval_result['loss'], data_type='dev')
                    if config.SAVE_USE_ACCURACY:
                        save_best_model(model_it_self, eval_result['accuracy'], data_type='dev',
                                        use_accuracy=config.SAVE_USE_ACCURACY)
    shutil.copy(config.train_best_accuracy_model, os.path.join(config.output_dir, 'best_ac_model.bin'))
    shutil.copy(config.train_best_loss_model, os.path.join(config.output_dir, 'best_loss_model.bin'))


####
# Name: init
# Function: 初始化
####
def init(model):
    if config.init_checkpoint is not None:
        state_dict = torch.load(config.init_checkpoint, map_location='cpu')
        new_keys = ["embeddings.word_embeddings.weight", "embeddings.position_embeddings.weight", "embeddings.token_type_embeddings.weight", "embeddings.LayerNorm.gamma", "embeddings.LayerNorm.beta", "encoder.layer.0.attention.self.query.weight", "encoder.layer.0.attention.self.query.bias", "encoder.layer.0.attention.self.key.weight", "encoder.layer.0.attention.self.key.bias", "encoder.layer.0.attention.self.value.weight", "encoder.layer.0.attention.self.value.bias", "encoder.layer.0.attention.output.dense.weight", "encoder.layer.0.attention.output.dense.bias", "encoder.layer.0.attention.output.LayerNorm.gamma", "encoder.layer.0.attention.output.LayerNorm.beta", "encoder.layer.0.intermediate.dense.weight", "encoder.layer.0.intermediate.dense.bias", "encoder.layer.0.output.dense.weight", "encoder.layer.0.output.dense.bias", "encoder.layer.0.output.LayerNorm.gamma", "encoder.layer.0.output.LayerNorm.beta", "encoder.layer.1.attention.self.query.weight", "encoder.layer.1.attention.self.query.bias", "encoder.layer.1.attention.self.key.weight", "encoder.layer.1.attention.self.key.bias", "encoder.layer.1.attention.self.value.weight", "encoder.layer.1.attention.self.value.bias", "encoder.layer.1.attention.output.dense.weight", "encoder.layer.1.attention.output.dense.bias", "encoder.layer.1.attention.output.LayerNorm.gamma", "encoder.layer.1.attention.output.LayerNorm.beta", "encoder.layer.1.intermediate.dense.weight", "encoder.layer.1.intermediate.dense.bias", "encoder.layer.1.output.dense.weight", "encoder.layer.1.output.dense.bias", "encoder.layer.1.output.LayerNorm.gamma", "encoder.layer.1.output.LayerNorm.beta", "encoder.layer.2.attention.self.query.weight", "encoder.layer.2.attention.self.query.bias", "encoder.layer.2.attention.self.key.weight", "encoder.layer.2.attention.self.key.bias", "encoder.layer.2.attention.self.value.weight", "encoder.layer.2.attention.self.value.bias", "encoder.layer.2.attention.output.dense.weight", "encoder.layer.2.attention.output.dense.bias", "encoder.layer.2.attention.output.LayerNorm.gamma", "encoder.layer.2.attention.output.LayerNorm.beta", "encoder.layer.2.intermediate.dense.weight", "encoder.layer.2.intermediate.dense.bias", "encoder.layer.2.output.dense.weight", "encoder.layer.2.output.dense.bias", "encoder.layer.2.output.LayerNorm.gamma", "encoder.layer.2.output.LayerNorm.beta", "encoder.layer.3.attention.self.query.weight", "encoder.layer.3.attention.self.query.bias", "encoder.layer.3.attention.self.key.weight", "encoder.layer.3.attention.self.key.bias", "encoder.layer.3.attention.self.value.weight", "encoder.layer.3.attention.self.value.bias", "encoder.layer.3.attention.output.dense.weight", "encoder.layer.3.attention.output.dense.bias", "encoder.layer.3.attention.output.LayerNorm.gamma", "encoder.layer.3.attention.output.LayerNorm.beta", "encoder.layer.3.intermediate.dense.weight", "encoder.layer.3.intermediate.dense.bias", "encoder.layer.3.output.dense.weight", "encoder.layer.3.output.dense.bias", "encoder.layer.3.output.LayerNorm.gamma", "encoder.layer.3.output.LayerNorm.beta", "encoder.layer.4.attention.self.query.weight", "encoder.layer.4.attention.self.query.bias", "encoder.layer.4.attention.self.key.weight", "encoder.layer.4.attention.self.key.bias", "encoder.layer.4.attention.self.value.weight", "encoder.layer.4.attention.self.value.bias", "encoder.layer.4.attention.output.dense.weight", "encoder.layer.4.attention.output.dense.bias", "encoder.layer.4.attention.output.LayerNorm.gamma", "encoder.layer.4.attention.output.LayerNorm.beta", "encoder.layer.4.intermediate.dense.weight", "encoder.layer.4.intermediate.dense.bias", "encoder.layer.4.output.dense.weight", "encoder.layer.4.output.dense.bias", "encoder.layer.4.output.LayerNorm.gamma", "encoder.layer.4.output.LayerNorm.beta", "encoder.layer.5.attention.self.query.weight", "encoder.layer.5.attention.self.query.bias", "encoder.layer.5.attention.self.key.weight", "encoder.layer.5.attention.self.key.bias", "encoder.layer.5.attention.self.value.weight", "encoder.layer.5.attention.self.value.bias", "encoder.layer.5.attention.output.dense.weight", "encoder.layer.5.attention.output.dense.bias", "encoder.layer.5.attention.output.LayerNorm.gamma", "encoder.layer.5.attention.output.LayerNorm.beta", "encoder.layer.5.intermediate.dense.weight", "encoder.layer.5.intermediate.dense.bias", "encoder.layer.5.output.dense.weight", "encoder.layer.5.output.dense.bias", "encoder.layer.5.output.LayerNorm.gamma", "encoder.layer.5.output.LayerNorm.beta", "encoder.layer.6.attention.self.query.weight", "encoder.layer.6.attention.self.query.bias", "encoder.layer.6.attention.self.key.weight", "encoder.layer.6.attention.self.key.bias", "encoder.layer.6.attention.self.value.weight", "encoder.layer.6.attention.self.value.bias", "encoder.layer.6.attention.output.dense.weight", "encoder.layer.6.attention.output.dense.bias", "encoder.layer.6.attention.output.LayerNorm.gamma", "encoder.layer.6.attention.output.LayerNorm.beta", "encoder.layer.6.intermediate.dense.weight", "encoder.layer.6.intermediate.dense.bias", "encoder.layer.6.output.dense.weight", "encoder.layer.6.output.dense.bias", "encoder.layer.6.output.LayerNorm.gamma", "encoder.layer.6.output.LayerNorm.beta", "encoder.layer.7.attention.self.query.weight", "encoder.layer.7.attention.self.query.bias", "encoder.layer.7.attention.self.key.weight", "encoder.layer.7.attention.self.key.bias", "encoder.layer.7.attention.self.value.weight", "encoder.layer.7.attention.self.value.bias", "encoder.layer.7.attention.output.dense.weight", "encoder.layer.7.attention.output.dense.bias", "encoder.layer.7.attention.output.LayerNorm.gamma", "encoder.layer.7.attention.output.LayerNorm.beta", "encoder.layer.7.intermediate.dense.weight", "encoder.layer.7.intermediate.dense.bias", "encoder.layer.7.output.dense.weight", "encoder.layer.7.output.dense.bias", "encoder.layer.7.output.LayerNorm.gamma", "encoder.layer.7.output.LayerNorm.beta", "encoder.layer.8.attention.self.query.weight", "encoder.layer.8.attention.self.query.bias", "encoder.layer.8.attention.self.key.weight", "encoder.layer.8.attention.self.key.bias", "encoder.layer.8.attention.self.value.weight", "encoder.layer.8.attention.self.value.bias", "encoder.layer.8.attention.output.dense.weight", "encoder.layer.8.attention.output.dense.bias", "encoder.layer.8.attention.output.LayerNorm.gamma", "encoder.layer.8.attention.output.LayerNorm.beta", "encoder.layer.8.intermediate.dense.weight", "encoder.layer.8.intermediate.dense.bias", "encoder.layer.8.output.dense.weight", "encoder.layer.8.output.dense.bias", "encoder.layer.8.output.LayerNorm.gamma", "encoder.layer.8.output.LayerNorm.beta", "encoder.layer.9.attention.self.query.weight", "encoder.layer.9.attention.self.query.bias", "encoder.layer.9.attention.self.key.weight", "encoder.layer.9.attention.self.key.bias", "encoder.layer.9.attention.self.value.weight", "encoder.layer.9.attention.self.value.bias", "encoder.layer.9.attention.output.dense.weight", "encoder.layer.9.attention.output.dense.bias", "encoder.layer.9.attention.output.LayerNorm.gamma", "encoder.layer.9.attention.output.LayerNorm.beta", "encoder.layer.9.intermediate.dense.weight", "encoder.layer.9.intermediate.dense.bias", "encoder.layer.9.output.dense.weight", "encoder.layer.9.output.dense.bias", "encoder.layer.9.output.LayerNorm.gamma", "encoder.layer.9.output.LayerNorm.beta", "encoder.layer.10.attention.self.query.weight", "encoder.layer.10.attention.self.query.bias", "encoder.layer.10.attention.self.key.weight", "encoder.layer.10.attention.self.key.bias", "encoder.layer.10.attention.self.value.weight", "encoder.layer.10.attention.self.value.bias", "encoder.layer.10.attention.output.dense.weight", "encoder.layer.10.attention.output.dense.bias", "encoder.layer.10.attention.output.LayerNorm.gamma", "encoder.layer.10.attention.output.LayerNorm.beta", "encoder.layer.10.intermediate.dense.weight", "encoder.layer.10.intermediate.dense.bias", "encoder.layer.10.output.dense.weight", "encoder.layer.10.output.dense.bias", "encoder.layer.10.output.LayerNorm.gamma", "encoder.layer.10.output.LayerNorm.beta", "encoder.layer.11.attention.self.query.weight", "encoder.layer.11.attention.self.query.bias", "encoder.layer.11.attention.self.key.weight", "encoder.layer.11.attention.self.key.bias", "encoder.layer.11.attention.self.value.weight", "encoder.layer.11.attention.self.value.bias", "encoder.layer.11.attention.output.dense.weight", "encoder.layer.11.attention.output.dense.bias", "encoder.layer.11.attention.output.LayerNorm.gamma", "encoder.layer.11.attention.output.LayerNorm.beta", "encoder.layer.11.intermediate.dense.weight", "encoder.layer.11.intermediate.dense.bias", "encoder.layer.11.output.dense.weight", "encoder.layer.11.output.dense.bias", "encoder.layer.11.output.LayerNorm.gamma", "encoder.layer.11.output.LayerNorm.beta", "pooler.dense.weight", "pooler.dense.bias"]
        old_keys = ["bert.embeddings.word_embeddings.weight", "bert.embeddings.position_embeddings.weight", "bert.embeddings.token_type_embeddings.weight", "bert.embeddings.LayerNorm.weight", "bert.embeddings.LayerNorm.bias", "bert.encoder.layer.0.attention.self.query.weight", "bert.encoder.layer.0.attention.self.query.bias", "bert.encoder.layer.0.attention.self.key.weight", "bert.encoder.layer.0.attention.self.key.bias", "bert.encoder.layer.0.attention.self.value.weight", "bert.encoder.layer.0.attention.self.value.bias", "bert.encoder.layer.0.attention.output.dense.weight", "bert.encoder.layer.0.attention.output.dense.bias", "bert.encoder.layer.0.attention.output.LayerNorm.weight", "bert.encoder.layer.0.attention.output.LayerNorm.bias", "bert.encoder.layer.0.intermediate.dense.weight", "bert.encoder.layer.0.intermediate.dense.bias", "bert.encoder.layer.0.output.dense.weight", "bert.encoder.layer.0.output.dense.bias", "bert.encoder.layer.0.output.LayerNorm.weight", "bert.encoder.layer.0.output.LayerNorm.bias", "bert.encoder.layer.1.attention.self.query.weight", "bert.encoder.layer.1.attention.self.query.bias", "bert.encoder.layer.1.attention.self.key.weight", "bert.encoder.layer.1.attention.self.key.bias", "bert.encoder.layer.1.attention.self.value.weight", "bert.encoder.layer.1.attention.self.value.bias", "bert.encoder.layer.1.attention.output.dense.weight", "bert.encoder.layer.1.attention.output.dense.bias", "bert.encoder.layer.1.attention.output.LayerNorm.weight", "bert.encoder.layer.1.attention.output.LayerNorm.bias", "bert.encoder.layer.1.intermediate.dense.weight", "bert.encoder.layer.1.intermediate.dense.bias", "bert.encoder.layer.1.output.dense.weight", "bert.encoder.layer.1.output.dense.bias", "bert.encoder.layer.1.output.LayerNorm.weight", "bert.encoder.layer.1.output.LayerNorm.bias", "bert.encoder.layer.2.attention.self.query.weight", "bert.encoder.layer.2.attention.self.query.bias", "bert.encoder.layer.2.attention.self.key.weight", "bert.encoder.layer.2.attention.self.key.bias", "bert.encoder.layer.2.attention.self.value.weight", "bert.encoder.layer.2.attention.self.value.bias", "bert.encoder.layer.2.attention.output.dense.weight", "bert.encoder.layer.2.attention.output.dense.bias", "bert.encoder.layer.2.attention.output.LayerNorm.weight", "bert.encoder.layer.2.attention.output.LayerNorm.bias", "bert.encoder.layer.2.intermediate.dense.weight", "bert.encoder.layer.2.intermediate.dense.bias", "bert.encoder.layer.2.output.dense.weight", "bert.encoder.layer.2.output.dense.bias", "bert.encoder.layer.2.output.LayerNorm.weight", "bert.encoder.layer.2.output.LayerNorm.bias", "bert.encoder.layer.3.attention.self.query.weight", "bert.encoder.layer.3.attention.self.query.bias", "bert.encoder.layer.3.attention.self.key.weight", "bert.encoder.layer.3.attention.self.key.bias", "bert.encoder.layer.3.attention.self.value.weight", "bert.encoder.layer.3.attention.self.value.bias", "bert.encoder.layer.3.attention.output.dense.weight", "bert.encoder.layer.3.attention.output.dense.bias", "bert.encoder.layer.3.attention.output.LayerNorm.weight", "bert.encoder.layer.3.attention.output.LayerNorm.bias", "bert.encoder.layer.3.intermediate.dense.weight", "bert.encoder.layer.3.intermediate.dense.bias", "bert.encoder.layer.3.output.dense.weight", "bert.encoder.layer.3.output.dense.bias", "bert.encoder.layer.3.output.LayerNorm.weight", "bert.encoder.layer.3.output.LayerNorm.bias", "bert.encoder.layer.4.attention.self.query.weight", "bert.encoder.layer.4.attention.self.query.bias", "bert.encoder.layer.4.attention.self.key.weight", "bert.encoder.layer.4.attention.self.key.bias", "bert.encoder.layer.4.attention.self.value.weight", "bert.encoder.layer.4.attention.self.value.bias", "bert.encoder.layer.4.attention.output.dense.weight", "bert.encoder.layer.4.attention.output.dense.bias", "bert.encoder.layer.4.attention.output.LayerNorm.weight", "bert.encoder.layer.4.attention.output.LayerNorm.bias", "bert.encoder.layer.4.intermediate.dense.weight", "bert.encoder.layer.4.intermediate.dense.bias", "bert.encoder.layer.4.output.dense.weight", "bert.encoder.layer.4.output.dense.bias", "bert.encoder.layer.4.output.LayerNorm.weight", "bert.encoder.layer.4.output.LayerNorm.bias", "bert.encoder.layer.5.attention.self.query.weight", "bert.encoder.layer.5.attention.self.query.bias", "bert.encoder.layer.5.attention.self.key.weight", "bert.encoder.layer.5.attention.self.key.bias", "bert.encoder.layer.5.attention.self.value.weight", "bert.encoder.layer.5.attention.self.value.bias", "bert.encoder.layer.5.attention.output.dense.weight", "bert.encoder.layer.5.attention.output.dense.bias", "bert.encoder.layer.5.attention.output.LayerNorm.weight", "bert.encoder.layer.5.attention.output.LayerNorm.bias", "bert.encoder.layer.5.intermediate.dense.weight", "bert.encoder.layer.5.intermediate.dense.bias", "bert.encoder.layer.5.output.dense.weight", "bert.encoder.layer.5.output.dense.bias", "bert.encoder.layer.5.output.LayerNorm.weight", "bert.encoder.layer.5.output.LayerNorm.bias", "bert.encoder.layer.6.attention.self.query.weight", "bert.encoder.layer.6.attention.self.query.bias", "bert.encoder.layer.6.attention.self.key.weight", "bert.encoder.layer.6.attention.self.key.bias", "bert.encoder.layer.6.attention.self.value.weight", "bert.encoder.layer.6.attention.self.value.bias", "bert.encoder.layer.6.attention.output.dense.weight", "bert.encoder.layer.6.attention.output.dense.bias", "bert.encoder.layer.6.attention.output.LayerNorm.weight", "bert.encoder.layer.6.attention.output.LayerNorm.bias", "bert.encoder.layer.6.intermediate.dense.weight", "bert.encoder.layer.6.intermediate.dense.bias", "bert.encoder.layer.6.output.dense.weight", "bert.encoder.layer.6.output.dense.bias", "bert.encoder.layer.6.output.LayerNorm.weight", "bert.encoder.layer.6.output.LayerNorm.bias", "bert.encoder.layer.7.attention.self.query.weight", "bert.encoder.layer.7.attention.self.query.bias", "bert.encoder.layer.7.attention.self.key.weight", "bert.encoder.layer.7.attention.self.key.bias", "bert.encoder.layer.7.attention.self.value.weight", "bert.encoder.layer.7.attention.self.value.bias", "bert.encoder.layer.7.attention.output.dense.weight", "bert.encoder.layer.7.attention.output.dense.bias", "bert.encoder.layer.7.attention.output.LayerNorm.weight", "bert.encoder.layer.7.attention.output.LayerNorm.bias", "bert.encoder.layer.7.intermediate.dense.weight", "bert.encoder.layer.7.intermediate.dense.bias", "bert.encoder.layer.7.output.dense.weight", "bert.encoder.layer.7.output.dense.bias", "bert.encoder.layer.7.output.LayerNorm.weight", "bert.encoder.layer.7.output.LayerNorm.bias", "bert.encoder.layer.8.attention.self.query.weight", "bert.encoder.layer.8.attention.self.query.bias", "bert.encoder.layer.8.attention.self.key.weight", "bert.encoder.layer.8.attention.self.key.bias", "bert.encoder.layer.8.attention.self.value.weight", "bert.encoder.layer.8.attention.self.value.bias", "bert.encoder.layer.8.attention.output.dense.weight", "bert.encoder.layer.8.attention.output.dense.bias", "bert.encoder.layer.8.attention.output.LayerNorm.weight", "bert.encoder.layer.8.attention.output.LayerNorm.bias", "bert.encoder.layer.8.intermediate.dense.weight", "bert.encoder.layer.8.intermediate.dense.bias", "bert.encoder.layer.8.output.dense.weight", "bert.encoder.layer.8.output.dense.bias", "bert.encoder.layer.8.output.LayerNorm.weight", "bert.encoder.layer.8.output.LayerNorm.bias", "bert.encoder.layer.9.attention.self.query.weight", "bert.encoder.layer.9.attention.self.query.bias", "bert.encoder.layer.9.attention.self.key.weight", "bert.encoder.layer.9.attention.self.key.bias", "bert.encoder.layer.9.attention.self.value.weight", "bert.encoder.layer.9.attention.self.value.bias", "bert.encoder.layer.9.attention.output.dense.weight", "bert.encoder.layer.9.attention.output.dense.bias", "bert.encoder.layer.9.attention.output.LayerNorm.weight", "bert.encoder.layer.9.attention.output.LayerNorm.bias", "bert.encoder.layer.9.intermediate.dense.weight", "bert.encoder.layer.9.intermediate.dense.bias", "bert.encoder.layer.9.output.dense.weight", "bert.encoder.layer.9.output.dense.bias", "bert.encoder.layer.9.output.LayerNorm.weight", "bert.encoder.layer.9.output.LayerNorm.bias", "bert.encoder.layer.10.attention.self.query.weight", "bert.encoder.layer.10.attention.self.query.bias", "bert.encoder.layer.10.attention.self.key.weight", "bert.encoder.layer.10.attention.self.key.bias", "bert.encoder.layer.10.attention.self.value.weight", "bert.encoder.layer.10.attention.self.value.bias", "bert.encoder.layer.10.attention.output.dense.weight", "bert.encoder.layer.10.attention.output.dense.bias", "bert.encoder.layer.10.attention.output.LayerNorm.weight", "bert.encoder.layer.10.attention.output.LayerNorm.bias", "bert.encoder.layer.10.intermediate.dense.weight", "bert.encoder.layer.10.intermediate.dense.bias", "bert.encoder.layer.10.output.dense.weight", "bert.encoder.layer.10.output.dense.bias", "bert.encoder.layer.10.output.LayerNorm.weight", "bert.encoder.layer.10.output.LayerNorm.bias", "bert.encoder.layer.11.attention.self.query.weight", "bert.encoder.layer.11.attention.self.query.bias", "bert.encoder.layer.11.attention.self.key.weight", "bert.encoder.layer.11.attention.self.key.bias", "bert.encoder.layer.11.attention.self.value.weight", "bert.encoder.layer.11.attention.self.value.bias", "bert.encoder.layer.11.attention.output.dense.weight", "bert.encoder.layer.11.attention.output.dense.bias", "bert.encoder.layer.11.attention.output.LayerNorm.weight", "bert.encoder.layer.11.attention.output.LayerNorm.bias", "bert.encoder.layer.11.intermediate.dense.weight", "bert.encoder.layer.11.intermediate.dense.bias", "bert.encoder.layer.11.output.dense.weight", "bert.encoder.layer.11.output.dense.bias", "bert.encoder.layer.11.output.LayerNorm.weight", "bert.encoder.layer.11.output.LayerNorm.bias", "bert.pooler.dense.weight", "bert.pooler.dense.bias", "cls.predictions.bias", "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias", "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.bias", "cls.predictions.decoder.weight", "cls.seq_relationship.weight", "cls.seq_relationship.bias"]
        for key in list(state_dict.keys()):
            if 'cls.' in key:
                state_dict.pop(key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        model.bert.load_state_dict(state_dict)


def eval_test(model):
    best_model_path = [os.path.join(config.output_dir, config.eval_best_accuracy_model),
                       os.path.join(config.output_dir, config.eval_best_loss_model)]
    for best_model in best_model_path:
        checkpoint = torch.load(best_model)
        model.load_state_dict(checkpoint['net'], strict=False)
        model = model.to(config.device)
        test_loader = data_generator.get_test_loader()
        print("\n********" + best_model + "********")
        model_eval(model, test_loader, data_type='test')
    pass


def main():
    # random.seed(config.seed)
    # np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    model_set = {
        "Discourage": Discourage,
        "DiscourageMask": DiscourageMask
    }
    start_time = time.time()
    os.makedirs(config.output_dir, exist_ok=True)
    args = len(data_generator.get_labels()), data_generator.get_num_train_steps(), -1
    model = model_set[config.MODEL_NAME](*args)
    model = model.to(config.device)
    init(model)
    if config.do_train:
        train(model)
    if config.do_test:
        eval_test(model)
    end_time = time.time()
    print("总计耗时：%d m" % int((end_time - start_time) / 60))
    pass


if __name__ == '__main__':
    main()
