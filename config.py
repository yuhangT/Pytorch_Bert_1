import torch
import os

is_pre = False
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 实现卡号匹配
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
do_train = True
# do_train = False
do_test = True
do_eval = True
# do_eval = False
task_name = 'Discourage'
MODEL_NAME = task_name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_batch_size = 8
test_batch_size = 8
dev_batch_size = 8
gradient_accumulation_steps = 1
num_train_epochs = 3
eval_interval = 1000
print_interval = 1
SAVE_USE_ACCURACY = True
seed = 42
do_lower_case = True
learning_rate = 2e-5
warmup_proportion = 0.1
vocab_file = '/home/fwl/LW/bert/uncased_L-12_H-768_A-12/vocab.txt'
if is_pre:
	vocab_file = '/home/yssong/LW/2020/bert_torch/pretrain_model/checkpoint-100/vocab.txt'

bert_config_file = '/home/fwl/LW/bert/uncased_L-12_H-768_A-12/bert_config.json'
if is_pre:
	bert_config_file = '/home/yssong/LW/2020/bert_torch/pretrain_model/checkpoint-100/config.json'

init_checkpoint = '/home/fwl/LW/bert/uncased_L-12_H-768_A-12/torch_bert_uncased.bin'
if is_pre:
	init_checkpoint = '/home/yssong/LW/2020/bert_torch/pretrain_model/checkpoint-100/pytorch_model.bin'

max_seq_length = 256
eval_best_loss = 999
eval_best_accuracy = 0
# eval_best_accuracy_model = '/home/yssong/LW/2020/bert_torch/output_checkpoints_bert_2/best_ac_model_static.bin'
# eval_best_loss_model = '/home/yssong/LW/2020/bert_torch/output_checkpoints_bert_2/best_loss_model_static.bin'
eval_best_accuracy_model = 'best_ac_model.bin'
eval_best_loss_model = 'best_loss_model.bin'
data_dir = './data'
output_dir = './output_bert'
if is_pre:
	output_bert = '../output_bert_pre'
# data_dir = '/home/yssong/LW/2020/bert_torch/discourse5_data'
# output_dir = './output_discourse5_data'
local_rank = 0
mlm_probability = 0.15

