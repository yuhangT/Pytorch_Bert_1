# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import config as user_config
import numpy as np
from sklearn.metrics import f1_score


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.8,
                attention_probs_dropout_prob=0.8,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):
    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])    

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTPooler(nn.Module):
    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers, pooled_output

class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.latent_dim = 2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + self.latent_dim, num_labels)
        self.alpha = nn.Linear(config.hidden_size, 1)
        self.w = nn.Linear(config.hidden_size, config.hidden_size)

        # self.mean = nn.functional.tanh(nn.Linear(config.hidden_size, self.latent_dim))
        self.mean = nn.Sequential(
            nn.Linear(config.hidden_size, 100),
            nn.Tanh(),
            nn.Linear(100, self.latent_dim)
        )

        # self.log_var = nn.Linear(config.hidden_size, self.latent_dim)

        self.log_var = nn.Sequential(
            nn.Linear(config.hidden_size, 100),
            nn.Tanh(),
            nn.Linear(100, self.latent_dim)
        )

        self.latent_classifier = nn.Linear(self.latent_dim, num_labels)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        all_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        # batch_num, vector_num, hidden_size = all_output.size()
        # pooled_output = all_output[-2][0]
        # all_output = all_output[-1]
        # print("pooled_output.size(): ", pooled_output.size())
        batch_size = pooled_output.size()[0]
        pooled_output = self.dropout(pooled_output)
        z_mean = self.mean(pooled_output)
        z_log_var = self.log_var(pooled_output)
        z_mean.to(torch.device("cuda", 0))
        z_mean = z_mean.cuda()
        z_log_var.to(torch.device("cuda", 0))
        z_log_var = z_log_var.cuda()
        # sampling = torch.normal(mean=torch.arange(0., 1.), std=torch.arange(1., 2.), out = torch.tensor([float(pooled_output.size[0]), float(self.latent_dim)]))
        sampling_sum = 0
        for i in range(13):
            sampling_sum += torch.randn((batch_size, self.latent_dim))
        sampling = sampling_sum / 13
        sampling.to(torch.device("cuda", 0))
        sampling = sampling.cuda()
        # print(sampling)
        sampling_result = torch.exp(0.5 * z_log_var) * sampling + z_mean

        fuse_vector = torch.cat([pooled_output, sampling_result], 1)


        # print(all_output.size())
        # all_output = self.dropout(all_output)
        # attention_weight = torch.nn.functional.softmax(self.alpha(all_output), dim=1)
        # # fuse_vector = torch.sum(attention_weight * torch.nn.functional.tanh(self.w(all_output)), dim=1)
        # fuse_vector = self.dropout(fuse_vector)


        logits = self.classifier(fuse_vector)




        if labels is not None:
            kl_loss = 1 + z_log_var - z_mean**2 - torch.exp(z_log_var)
            kl_loss = torch.sum(kl_loss, dim=-1)
            kl_loss *= -0.5
            # vae_loss = K.mean(reconstruction_loss + kl_loss)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels) + torch.mean(kl_loss.float() )
            return loss, logits
        else:
            return logits


class Discourage(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, *arg):
        super(Discourage, self).__init__()
        self.result = {}
        num_labels, total_steps, _ = arg
        config = BertConfig.from_json_file(user_config.bert_config_file)
        self.bert = BertModel(config)
        #self.bert1 =BertModel(config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.labels_data = None
        
        #self.cls = BertOnlyMLMHead(config)
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(self, data, *arg):
        labels = None
        # print(len(data))
        if len(data) == 4:
            input_ids, attention_mask, token_type_ids, labels = data
        else:
            input_ids, attention_mask, token_type_ids = data
        if arg is not ():
            self.global_step, _ = arg
        self.labels_data = labels
        pooled_output = self.encoder(input_ids, attention_mask, token_type_ids)#bert参数跟新
        #pooled_output1 =self.encoder1(input_ids, attention_mask, token_type_ids)#bert参数不更新
        #pooled_output =self.w*pooled_output+(1-self.w)*pooled_output1
        output = self.decoder(pooled_output)
        if labels is not None:
            loss = self.calc_loss(output, labels)
            return loss, torch.nn.functional.softmax(output, dim=-1)
        return torch.nn.functional.softmax(output, dim=-1)

    def encoder(self, *x):
        input_ids, attention_mask, token_type_ids = x
        all_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        return pooled_output
    """
    def encoder1(self, *x):
        input_ids, attention_mask, token_type_ids = x
        all_output, pooled_output = self.bert1(input_ids, token_type_ids, attention_mask)
        return pooled_output
        """

    def decoder(self, x):
       # x = self.dropout(x)
        return self.classifier(x)

    def calc_loss(self, inputs, targets):
        # targets = targets.view(-1, 1)
        # print(inputs.shape)
        # print(targets.shape)
        loss_cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        loss_cross_entropy = loss_cross_entropy(inputs, targets)
        self.result['loss'] = loss_cross_entropy.detach().item()
        self.result['accuracy'] = self.calc_accuracy(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        self.result['f1'] = self.calc_f1(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        return loss_cross_entropy

    def calc_accuracy(self, inputs, targets):
        # inputs = (n, m)
        # target = (n, 1)
        outputs = np.argmax(inputs, axis=1)
        return np.mean(outputs == targets)

    def calc_f1(self, inputs, targets):
        # inputs = (n, m)
        # target = (n, 1)
        outputs = np.argmax(inputs, axis=1)
        return f1_score(targets, outputs, average='macro')

    def get_result(self):
        return self.result

    def get_labels_data(self):
        return self.labels_data



BertLayerNorm = torch.nn.LayerNorm
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class DiscourageMask(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, *arg):
        super(DiscourageMask, self).__init__()
        num_labels, total_steps, _ = arg
        self.total_steps = total_steps
        self.result = {}
        config = BertConfig.from_json_file(user_config.bert_config_file)
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.labels_data = None
        self.cls = BertOnlyMLMHead(config)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()


        self.apply(init_weights)

    def forward(self, data, *arg):
        labels = None
        masked_lm_labels = None
        if len(data) == 5:
            input_ids, attention_mask, token_type_ids, labels, masked_lm_labels = data
        elif len(data) == 4:
            input_ids, attention_mask, token_type_ids, labels = data
        else:
            input_ids, attention_mask, token_type_ids = data
        if arg is not ():
            self.global_step, _ = arg
        self.labels_data = labels
        all_output, pooled_output = self.encoder(input_ids, attention_mask, token_type_ids)
        #print(all_output[-1].shape)
        #print("input_ids, ", input_ids.shape)
        #print("token_type_ids: ", masked_lm_labels.shape)
        prediction_scores = self.cls(all_output[-1])
        output = self.decoder(pooled_output)
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits, labels)
        #     return loss, logits
        # else:
        #     return logits
        if labels is not None and masked_lm_labels is not None:
            loss = self.calc_loss(output, labels, prediction_scores, masked_lm_labels)

            return loss, torch.nn.functional.softmax(output, dim=-1)
        return torch.nn.functional.softmax(output, dim=-1)

    def encoder(self, *x):
        input_ids, attention_mask, token_type_ids = x
        # print(input_ids[0])
        # print(token_type_ids[0])
        # print(attention_mask[0])
        all_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        return all_output, pooled_output

    def decoder(self, x):
        return self.classifier(x)

    def calc_loss(self, inputs, targets, prediction_scores, masked_lm_labels):
        # targets = targets.view(-1, 1)
        # print(inputs.shape)
        # print(targets.shape)
        loss_cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        loss_cl = loss_cross_entropy(inputs, targets)
        loss_mask = loss_cross_entropy(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        loss = loss_cl
        # if self.global_step > self.total_steps:
        #     loss = loss_cl
        # else :
        #     loss = loss_cl + loss_mask
        self.result['loss'] = loss.detach().item()
        self.result['accuracy'] = self.calc_accuracy(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        self.result['f1'] = self.calc_f1(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        return loss

    def calc_accuracy(self, inputs, targets):
        # inputs = (n, m)
        # target = (n, 1)
        outputs = np.argmax(inputs, axis=1)
        return np.mean(outputs == targets)

    def calc_f1(self, inputs, targets):
        # inputs = (n, m)
        # target = (n, 1)
        outputs = np.argmax(inputs, axis=1)
        return f1_score(targets, outputs, average='macro')

    def get_result(self):
        return self.result

    def get_labels_data(self):
        return self.labels_data

#DiscourageMask(4)

