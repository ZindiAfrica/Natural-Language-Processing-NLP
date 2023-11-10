import copy
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.m2m_100.modeling_m2m_100 import (
    _CONFIG_FOR_DOC,
    M2M_100_START_DOCSTRING,
    M2M100Config,
    M2M100Encoder,
    M2M100PreTrainedModel,
)
from transformers.utils import add_start_docstrings, replace_return_docstrings
from transformers.utils.logging import get_logger

logger = get_logger("transformers")


@add_start_docstrings(
    """M2M100 Model with a token classification head on top
    (a linear layer on top of the hidden-states output)
    e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    M2M_100_START_DOCSTRING,
)
class M2M100ForTokenClassification(M2M100PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, config: M2M100Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        self.encoder = M2M100Encoder(config, self.shared)

        classifier_dropout = (
            config.classifier_dropout
            if hasattr(config, "classifier_dropout")
            else config.dropout
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.num_labels), labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
