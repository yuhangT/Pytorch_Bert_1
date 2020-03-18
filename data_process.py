import os
import pickle
from abc import ABC

import tokenization_word as tokenization
import logging
import config
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import random
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        file_in = open(input_file, "rb")
        lines = []
        for line in file_in:
            lines.append(line.decode("utf-8").split("\t"))
        return lines


class DiscourageProcessor(DataProcessor):
    """Processor for the Discourage data set ."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        tmp = list(self.labels)
        tmp.sort()
        return tmp

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[1])
            self.labels.add(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    # def _create_predict_examples(self, lines, set_type):
    #     """Creates examples for the training and dev sets."""
    #     examples = []
    #     for (i, line) in enumerate(lines):
    #         guid = "%s-%s" % (set_type, i)
    #         text_a = tokenization.convert_to_unicode(line[0])
    #         label = tokenization.convert_to_unicode(line[0])
    #         self.labels.add(label)
    #         examples.append(
    #             InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    #     return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class DataGenerate(object):

    def get_train_loader(self):
        raise NotImplementedError()

    def get_dev_loader(self):
        raise NotImplementedError()

    def get_test_loader(self):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()


####
# 按这个顺序 all_input_ids, all_input_mask, all_segment_ids, all_label_ids 生成数据
####
class DiscourageDataGenerate(DataGenerate):
    def __init__(self):
        processors = {
            "Discourage": DiscourageProcessor,
            "DiscourageTest": DiscourageProcessor
        }
        self.processor = processors[config.task_name]()
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=config.vocab_file, do_lower_case=config.do_lower_case)
        self.label_path = os.path.join(config.data_dir, "args.bin")
        self.train_path = os.path.join(config.data_dir, "train.bin")
        self.dev_path = os.path.join(config.data_dir, "dev.bin")
        self.test_path = os.path.join(config.data_dir, "test.bin")
        self.train_examples = None
        self.num_train_steps = None
        if not os.path.exists(self.label_path):
            self.train_examples = self.processor.get_train_examples(config.data_dir)
            label_list = self.processor.get_labels()
            self.fw = open(self.label_path, "wb+")
            pickle.dump(label_list, self.fw, -1)
            pickle.dump(len(self.train_examples), self.fw, -1)
            # self.fw.write(str(label_list))
            self.labels_list = label_list
            self.num_train_steps = len(self.train_examples)
        else:
            self.fw = open(self.label_path, "rb")
            self.labels_list = pickle.load(self.fw)
            self.num_train_steps = pickle.load(self.fw)
        self.fw.close()
        self.test_data_loader = None
        self.train_data_loader = None
        pass

    def get_train_loader(self):
        if self.train_data_loader is None:
            if not os.path.exists(self.train_path):
                if self.train_examples is None:
                    self.train_examples = self.processor.get_train_examples(config.data_dir)
                label_list = self.processor.get_labels()
                # with open(self.label_path, "w+", encoding='UTF-8') as fw:
                #     fw.write(str(label_list))
                self.labels_list = label_list
                train_features = convert_examples_to_features(
                    self.train_examples, self.get_labels(), config.max_seq_length, self.tokenizer)
                self.num_train_steps = int(
                    len(self.train_examples) / config.train_batch_size
                    / config.gradient_accumulation_steps * config.num_train_epochs)
                logger.info("***** Running training *****")
                logger.info("  Num examples = %d", len(self.train_examples))
                logger.info("  Batch size = %d", config.train_batch_size)
                logger.info("  Num steps = %d", self.num_train_steps)
                all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
                train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                with open(self.train_path, 'wb') as f:
                    pickle.dump(train_data, f, -1)
            else:
                with open(self.train_path, 'rb') as f:
                    train_data = pickle.load(f)
            train_sampler = RandomSampler(train_data)
            self.train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=config.train_batch_size)
        return self.train_data_loader

    def get_dev_loader(self):
        if not os.path.exists(self.dev_path):
            dev_examples = self.processor.get_dev_examples(config.data_dir)
            dev_features = convert_examples_to_features(
                dev_examples, self.get_labels(), config.max_seq_length, self.tokenizer)
            all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
            dev_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            
            with open(self.dev_path, 'wb') as f:
                pickle.dump(dev_data, f, -1)
        else:
            with open(self.dev_path, 'rb') as f:
                dev_data = pickle.load(f)
        dev_data_loader = DataLoader(dev_data, batch_size=config.dev_batch_size)
        return dev_data_loader

    def get_test_loader(self):
        if self.test_data_loader is None:
            if not os.path.exists(self.test_path):
                test_examples = self.processor.get_test_examples(config.data_dir)
                test_features = convert_examples_to_features(
                    test_examples, self.get_labels(), config.max_seq_length, self.tokenizer)
                all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
                test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                
                with open(self.test_path, 'wb') as f:
                    pickle.dump(test_data, f, -1)
            else:
                with open(self.test_path, 'rb') as f:
                    test_data = pickle.load(f)
            self.test_data_loader = DataLoader(test_data, batch_size=config.test_batch_size)
        return self.test_data_loader

    def get_labels(self):
        return self.labels_list

    def get_num_train_steps(self):
        return self.num_train_steps


####
# 按这个顺序 all_input_ids, all_input_mask, all_segment_ids, all_label_ids 生成数据
####
class DiscourageMaskDataGenerate(DataGenerate):
    def __init__(self):
        processors = {
            "Discourage": DiscourageProcessor,
            "DiscourageTest": DiscourageProcessor,
            "DiscourageMask": DiscourageProcessor
        }
        self.processor = processors[config.task_name]()
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=config.vocab_file, do_lower_case=config.do_lower_case)
        self.label_path = os.path.join(config.data_dir, "args.bin")
        self.train_path = os.path.join(config.data_dir, "train.bin")
        self.dev_path = os.path.join(config.data_dir, "dev.bin")
        self.test_path = os.path.join(config.data_dir, "test.bin")
        self.train_examples = None
        self.num_train_steps = None
        if not os.path.exists(self.label_path):
            self.train_examples = self.processor.get_train_examples(config.data_dir)
            label_list = self.processor.get_labels()
            self.fw = open(self.label_path, "wb+")
            pickle.dump(label_list, self.fw, -1)
            pickle.dump(len(self.train_examples), self.fw, -1)
            # self.fw.write(str(label_list))
            self.labels_list = label_list
            self.num_train_steps = len(self.train_examples)
        else:
            self.fw = open(self.label_path, "rb")
            self.labels_list = pickle.load(self.fw)
            self.num_train_steps = pickle.load(self.fw)
        self.fw.close()
        self.test_data_loader = None
        self.train_data_loader = None
        pass

    def getbool(self, t):
        return torch.tensor(t.numpy() > 0.1)

    def mask_tokens(self, inputs: torch.Tensor):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, config.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        print(special_tokens_mask)
        probability_matrix.masked_fill_(torch.ByteTensor(special_tokens_mask), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).byte()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).byte() & masked_indices
        inputs[indices_replaced] = 103

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).byte() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def get_train_loader(self):
        if self.train_data_loader is None:
            if not os.path.exists(self.train_path):
                if self.train_examples is None:
                    self.train_examples = self.processor.get_train_examples(config.data_dir)
                label_list = self.processor.get_labels()
                # with open(self.label_path, "w+", encoding='UTF-8') as fw:
                #     fw.write(str(label_list))
                self.labels_list = label_list
                train_features = convert_examples_to_features(
                    self.train_examples, self.get_labels(), config.max_seq_length, self.tokenizer)
                self.num_train_steps = int(
                    len(self.train_examples) / config.train_batch_size
                    / config.gradient_accumulation_steps * config.num_train_epochs)
                logger.info("***** Running training *****")
                logger.info("  Num examples = %d", len(self.train_examples))
                logger.info("  Batch size = %d", config.train_batch_size)
                logger.info("  Num steps = %d", self.num_train_steps)
                all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
                all_input_ids, mask_labels = self.mask_tokens(all_input_ids)
                all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
                train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, mask_labels)
                
                with open(self.train_path, 'wb') as f:
                    pickle.dump(train_data, f, -1)
            else:
                with open(self.train_path, 'rb') as f:
                    train_data = pickle.load(f)
            train_sampler = SequentialSampler(train_data)
            self.train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=config.train_batch_size)
        return self.train_data_loader

    def get_dev_loader(self):
        if not os.path.exists(self.dev_path):
            dev_examples = self.processor.get_dev_examples(config.data_dir)
            dev_features = convert_examples_to_features(
                dev_examples, self.get_labels(), config.max_seq_length, self.tokenizer)
            all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
            all_input_ids, mask_labels = self.mask_tokens(all_input_ids)
            all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
            dev_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, mask_labels)
            dev_data_loader = DataLoader(dev_data, batch_size=config.dev_batch_size)
            with open(self.dev_path, 'wb') as f:
                pickle.dump(dev_data_loader, f, -1)
        else:
            with open(self.dev_path, 'rb') as f:
                dev_data_loader = pickle.load(f)
        return dev_data_loader

    def get_test_loader(self):
        if self.test_data_loader is None:
            if not os.path.exists(self.test_path):
                test_examples = self.processor.get_test_examples(config.data_dir)
                test_features = convert_examples_to_features(
                    test_examples, self.get_labels(), config.max_seq_length, self.tokenizer)
                all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
                all_input_ids, mask_labels = self.mask_tokens(all_input_ids)
                all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
                test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, mask_labels)
                self.test_data_loader = DataLoader(test_data, batch_size=config.test_batch_size)
                with open(self.test_path, 'wb') as f:
                    pickle.dump(self.test_data_loader, f, -1)
            else:
                with open(self.test_path, 'rb') as f:
                    self.test_data_loader = pickle.load(f)
        return self.test_data_loader

    def get_labels(self):
        return self.labels_list

    def get_num_train_steps(self):
        return self.num_train_steps


random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
data_generators = {
    "Discourage": DiscourageDataGenerate,
    "DiscourageTest": DiscourageDataGenerate,
    "DiscourageMask": DiscourageMaskDataGenerate
}
data_generator = data_generators[config.task_name]()
# print(config.task_name, data_generator.get_labels())
# train_loader = data_generator.get_train_loader()
# for x in train_loader:
#    print(x[0][0])
#    break
# for x in train_loader:
#    print(x[0][0])
#    break
# #

# print(data_generator.get_labels())
# class DiscourageGenerate(object):
