# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertModel


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertPooler(nn.Module):
    def __init__(self):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertIntermediate(nn.Module):
    def __init__(self):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(768, 3072)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class BertCoAttention(nn.Module):
    def __init__(self):
        super(BertCoAttention, self).__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        # s2_attention_mask  b*1*1*49
        mixed_query_layer = self.query(s1_hidden_states)  # b*75*768
        mixed_key_layer = self.key(s2_hidden_states)  # b*49*768
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # b*12*75*64
        key_layer = self.transpose_for_scores(mixed_key_layer)  # b*12*49*64
        value_layer = self.transpose_for_scores(mixed_value_layer)  # b*12*49*64

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # b*12*75*49
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # b*12*75*49
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask
        # atention_scores b*12*75*49
        # Normalize the attention scores to probabilities.
        # b*12*75*49
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer b*12*75*64
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer b*75*768
        return context_layer


class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfOutput(nn.Module):
    def __init__(self):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossAttention(nn.Module):
    def __init__(self):
        super(BertCrossAttention, self).__init__()
        self.bertCoAttn = BertCoAttention()
        self.output = BertSelfOutput()

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.bertCoAttn(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self):
        super(BertCrossAttentionLayer, self).__init__()
        self.bertCorssAttn = BertCrossAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.bertCorssAttn(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        # b*75*768
        intermediate_output = self.intermediate(attention_output)
        # b*75*3072
        layer_output = self.output(intermediate_output, attention_output)
        # b*75*3072
        return layer_output


class BertCrossEncoder(nn.Module):
    def __init__(self):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(3)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        return s1_hidden_states


class MsdBERT(nn.Module):
    def __init__(self):
        super(MsdBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hashtag_bert = BertModel.from_pretrained('bert-base-uncased')
        self.tanh = nn.Tanh()
        self.text2image_attention = BertCrossEncoder()
        self.image_text_pooler = BertPooler()
        self.dropout = nn.Dropout(0.1)
        self.vismap2text = nn.Linear(2048, 768)
        self.classifier = nn.Linear(768 * 2, 2)
        self.W_b = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(768, 768)))

    def forward(self, input_ids, visual_embeds_att, input_mask, added_attention_mask, hashtag_input_ids,
                hashtag_input_mask, labels=None):
        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)
        hashtag_output, hashtag_pooled_output = self.hashtag_bert(input_ids=hashtag_input_ids, token_type_ids=None,
                                                                  attention_mask=hashtag_input_mask)
        # added_attention_mask batch_size*124
        # img_mask # b*49
        img_mask = added_attention_mask[:, :49]
        # img_mask # b*1*1*49
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0
        # text作为query， image作为key和value

        # batchsize*49*2048
        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)
        # b*49*768
        visual = self.vismap2text(vis_embed_map)
        # b*75*768
        image_text_cross_attn = self.text2image_attention(sequence_output, visual, extended_img_mask)
        # b*75*12
        C = self.tanh(torch.matmul(torch.matmul(sequence_output, self.W_b), hashtag_output.transpose(1,2)))
        # C: b*12
        C, _ = torch.max(C, dim=1)
        attn = F.softmax(C, dim=-1)
        # b*1*768
        hashtag_text_cross_attn = torch.matmul(attn.unsqueeze(1), hashtag_output)
        # b*1*768
        image_text_pooled_output = self.image_text_pooler(image_text_cross_attn)
        pooled_output = torch.cat([image_text_pooled_output, hashtag_text_cross_attn.squeeze(1)], dim=-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits


class Res_BERT(nn.Module):
    def __init__(self):
        super(Res_BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        self.vismap2text = nn.Linear(2048, 768)

    def forward(self, input_ids, visual_embeds_att, input_mask, added_attention_mask, hashtag_input_ids,
                hashtag_input_mask, labels=None):
        # b*75*768
        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)
        # batchsize*49*2048
        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)
        # b*49*768
        visual = self.vismap2text(vis_embed_map)
        # b*1*768
        res = torch.cat([sequence_output, visual], dim=1).mean(1)
        pooled_output = self.dropout(res)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits


class BertOnly(nn.Module):
    def __init__(self):
        super(BertOnly, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, visual_embeds_att, input_mask, added_attention_mask, hashtag_input_ids,
                hashtag_input_mask, labels=None):
        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits


class ResNetOnly(nn.Module):
    def __init__(self):
        super(ResNetOnly, self).__init__()
        self.vismap2text = nn.Linear(2048, 768)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, visual_embeds_att, input_mask, added_attention_mask, hashtag_input_ids,
                hashtag_input_mask, labels=None):
        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)
        # b*49*2048
        vis_embed_map = self.vismap2text(vis_embed_map)
        vis_embed_map = vis_embed_map.mean(1)
        logits = self.classifier(vis_embed_map)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits
