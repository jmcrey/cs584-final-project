#!/usr/bin/env python
# coding: utf-8

# # Instructions:
# 
# 1. Run the wget command below to download the attention model
# 2. Untar the downloaded file and modify the `config.json` file
#     - Change the key `model.type` to "textual_entailment"
# 3. Retar the downloaded file and name it "textual_entailment.tar.gz"
# 4. Run the next two lines of code to make sure everything worked


######################################################################
######################################################################
######################################################################
#### This is the exact code from the AllenNLP Models library. The ####
#### only difference is an additional output called               ####
#### "aggregate_inputs", which is the input right before the      ####
#### softmax function. I am including this because it will be the ####
#### input to the MLP. Find the original source code link below.  ####
#### All credit goes to the original developers.                  ####
######################################################################
######################################################################
######################################################################

# Link: https://github.com/allenai/allennlp-models/blob/main/allennlp_models/pair_classification/models/decomposable_attention.py

from typing import List, Dict, Any
import torch

from allennlp_models.pair_classification.models import DecomposableAttention
from allennlp.data import TextFieldTensors
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum

@Model.register("textual_entailment")
class TextualEntailment(DecomposableAttention):
    
    def forward(  # type: ignore
        self,
        premise: TextFieldTensors,
        hypothesis: TextFieldTensors,
        label: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        premise : `TextFieldTensors`
            From a `TextField`
        hypothesis : `TextFieldTensors`
            From a `TextField`
        label : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`
        metadata : `List[Dict[str, Any]]`, optional (default = `None`)
            Metadata containing the original tokenization of the premise and
            hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.
        # Returns
        An output dictionary consisting of:
        label_logits : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_labels)` representing unnormalised log
            probabilities of the entailment label.
        label_probs : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_labels)` representing probabilities of the
            entailment label.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        """
        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        premise_mask = get_text_field_mask(premise)
        hypothesis_mask = get_text_field_mask(hypothesis)

        if self._premise_encoder:
            embedded_premise = self._premise_encoder(embedded_premise, premise_mask)
        if self._hypothesis_encoder:
            embedded_hypothesis = self._hypothesis_encoder(embedded_hypothesis, hypothesis_mask)

        projected_premise = self._attend_feedforward(embedded_premise)
        projected_hypothesis = self._attend_feedforward(embedded_hypothesis)
        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(projected_premise, projected_hypothesis)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = masked_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(embedded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(embedded_premise, h2p_attention)

        premise_compare_input = torch.cat([embedded_premise, attended_hypothesis], dim=-1)
        hypothesis_compare_input = torch.cat([embedded_hypothesis, attended_premise], dim=-1)

        compared_premise = self._compare_feedforward(premise_compare_input)
        compared_premise = compared_premise * premise_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_premise = compared_premise.sum(dim=1)

        compared_hypothesis = self._compare_feedforward(hypothesis_compare_input)
        compared_hypothesis = compared_hypothesis * hypothesis_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_hypothesis = compared_hypothesis.sum(dim=1)

        aggregate_input = torch.cat([compared_premise, compared_hypothesis], dim=-1)
        label_logits = self._aggregate_feedforward(aggregate_input)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {
            "label_logits": label_logits,
            "label_probs": label_probs,
            "aggregate_input": aggregate_input,
            "h2p_attention": h2p_attention,
            "p2h_attention": p2h_attention,
        }

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["premise_tokens"] = [x["premise_tokens"] for x in metadata]
            output_dict["hypothesis_tokens"] = [x["hypothesis_tokens"] for x in metadata]

        return output_dict
