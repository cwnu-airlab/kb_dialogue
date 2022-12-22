# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import transformers
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils_fast import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_utils import PreTrainedModel,apply_chunking_to_forward
from transformers.utils.doc import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import (
        BaseModelOutputWithPastAndCrossAttentions,
        BaseModelOutputWithPoolingAndCrossAttentions,
        MaskedLMOutput,
        ModelOutput,
)
from huggingface_hub import hf_hub_download
from transformers.activations import ACT2FN
import os
import torch, math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch import nn
import torch.nn.functional as F
from packaging import version

_import_structure = {
    "configuration_realm": ["REALM_PRETRAINED_CONFIG_ARCHIVE_MAP", "RealmConfig"],
    "tokenization_realm": ["RealmTokenizer"],
}

class RealmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of

    1. [`RealmEmbedder`]
    2. [`RealmScorer`]
    3. [`RealmKnowledgeAugEncoder`]
    5. [`RealmReader`]
    6. [`RealmForOpenQA`]

    It is used to instantiate an REALM model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the REALM
    [realm-cc-news-pretrained](https://huggingface.co/google/realm-cc-news-pretrained-embedder) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the REALM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RealmEmbedder`], [`RealmScorer`], [`RealmKnowledgeAugEncoder`], or
            [`RealmReader`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        retriever_proj_size (`int`, *optional*, defaults to 128):
            Dimension of the retriever(embedder) projection.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_candidates (`int`, *optional*, defaults to 8):
            Number of candidates inputted to the RealmScorer or RealmKnowledgeAugEncoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`RealmEmbedder`], [`RealmScorer`],
            [`RealmKnowledgeAugEncoder`], or [`RealmReader`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        span_hidden_size (`int`, *optional*, defaults to 256):
            Dimension of the reader's spans.
        max_span_width (`int`, *optional*, defaults to 10):
            Max span width of the reader.
        reader_layer_norm_eps (`float`, *optional*, defaults to 1e-3):
            The epsilon used by the reader's layer normalization layers.
        reader_beam_size (`int`, *optional*, defaults to 5):
            Beam size of the reader.
        reader_seq_len (`int`, *optional*, defaults to 288+32):
            Maximum sequence length of the reader.
        num_block_records (`int`, *optional*, defaults to 13353718):
            Number of block records.
        searcher_beam_size (`int`, *optional*, defaults to 5000):
            Beam size of the searcher. Note that when eval mode is enabled, *searcher_beam_size* will be the same as
            *reader_beam_size*.
        searcher_seq_len (`int`, *optional*, defaults to 64):
            Maximum sequence length of the searcher.

    Example:

    ```python
    >>> from transformers import RealmEmbedder, RealmConfig

    >>> # Initializing a REALM realm-cc-news-pretrained-* style configuration
    >>> configuration = RealmConfig()

    >>> # Initializing a model from the google/realm-cc-news-pretrained-embedder style configuration
    >>> model = RealmEmbedder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "realm"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        retriever_proj_size=128,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_candidates=8,
        intermediate_size=3072,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        span_hidden_size=256,
        max_span_width=10,
        reader_layer_norm_eps=1e-3,
        reader_beam_size=5,
        reader_seq_len=320,  # 288 + 32
        num_block_records=13353718,
        searcher_beam_size=5000,
        searcher_seq_len=64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # Common config
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.retriever_proj_size = retriever_proj_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_candidates = num_candidates
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps

        # Reader config
        self.span_hidden_size = span_hidden_size
        self.max_span_width = max_span_width
        self.reader_layer_norm_eps = reader_layer_norm_eps
        self.reader_beam_size = reader_beam_size
        self.reader_seq_len = reader_seq_len

        # Retrieval config
        self.num_block_records = num_block_records
        self.searcher_beam_size = searcher_beam_size
        self.searcher_seq_len = searcher_seq_len

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

def load_tf_weights_in_realm(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        if isinstance(model, RealmReader) and "reader" not in name:
            logger.info(f"Skipping {name} as it is not {model.__class__.__name__}'s parameter")
            continue

        # For pretrained openqa reader
        if (name.startswith("bert") or name.startswith("cls")) and isinstance(model, RealmForOpenQA):
            name = name.replace("bert/", "reader/realm/")
            name = name.replace("cls/", "reader/cls/")

        # For pretrained encoder
        if (name.startswith("bert") or name.startswith("cls")) and isinstance(model, RealmKnowledgeAugEncoder):
            name = name.replace("bert/", "realm/")

        # For finetuned reader
        if name.startswith("reader"):
            reader_prefix = "" if isinstance(model, RealmReader) else "reader/"
            name = name.replace("reader/module/bert/", f"{reader_prefix}realm/")
            name = name.replace("reader/module/cls/", f"{reader_prefix}cls/")
            name = name.replace("reader/dense/", f"{reader_prefix}qa_outputs/dense_intermediate/")
            name = name.replace("reader/dense_1/", f"{reader_prefix}qa_outputs/dense_output/")
            name = name.replace("reader/layer_normalization", f"{reader_prefix}qa_outputs/layer_normalization")

        # For embedder and scorer
        if name.startswith("module/module/module/"):  # finetuned
            embedder_prefix = "" if isinstance(model, RealmEmbedder) else "embedder/"
            name = name.replace("module/module/module/module/bert/", f"{embedder_prefix}realm/")
            name = name.replace("module/module/module/LayerNorm/", f"{embedder_prefix}cls/LayerNorm/")
            name = name.replace("module/module/module/dense/", f"{embedder_prefix}cls/dense/")
            name = name.replace("module/module/module/module/cls/predictions/", f"{embedder_prefix}cls/predictions/")
            name = name.replace("module/module/module/bert/", f"{embedder_prefix}realm/")
            name = name.replace("module/module/module/cls/predictions/", f"{embedder_prefix}cls/predictions/")
        elif name.startswith("module/module/"):  # pretrained
            embedder_prefix = "" if isinstance(model, RealmEmbedder) else "embedder/"
            name = name.replace("module/module/LayerNorm/", f"{embedder_prefix}cls/LayerNorm/")
            name = name.replace("module/module/dense/", f"{embedder_prefix}cls/dense/")

        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model

class M3ScorerProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.projected_size)
        self.LayerNorm = nn.LayerNorm(config.projected_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class M3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RealmConfig
    load_tf_weights = load_tf_weights_in_realm
    base_model_prefix = "realm"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _flatten_inputs(self, *inputs):
        """Flatten inputs' shape to (-1, input_shape[-1])"""
        flattened_inputs = []
        for tensor in inputs:
            if tensor is None:
                flattened_inputs.append(None)
            else:
                input_shape = tensor.shape
                if len(input_shape) > 2:
                    tensor = tensor.view((-1, input_shape[-1]))
                flattened_inputs.append(tensor)
        return flattened_inputs

@dataclass
class M3RetrieverOutput(ModelOutput):
    retrieved_logits : torch.FloatTensor = None
    has_answers : Optional[bool] = None
    concat_inputs : torch.FloatTensor = None
    retrieved_block_ids : torch.Tensor = None
    labels : torch.FloatTensor = None

class M3Retriever(M3PreTrainedModel):
    def __init__(self, config, tokenizer=None, block_records=None, block_emb=None, labels=None):
        super().__init__(config)

        self.tokenizer = tokenizer
        self.block_records = block_records
        self.block_emb = block_emb
        self.labels = labels
        concat_max_length = self.config.max_knowledge_length+self.config.max_source_length
        self.concat_max_length =concat_max_length if concat_max_length <=512 else 512
        self.post_init()

    @property
    def searcher_beam_size(self):
        if self.training:
            return self.config.searcher_beam_size
        return self.config.reader_beam_size

    def block_embedding_to(self, device):
        """Send `self.block_emb` to a specific device.

        Args:
            device (`str` or `torch.device`):
                The device to which `self.block_emb` will be sent.
        """

        self.block_emb = self.block_emb.to(device)

    def batched_index_select(input, dim, index):
        views = [1 if i != dim else -1 for i in range(len(input.shape))]
        expanse = list(input.shape)
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        # making the first dim of output be B
        return torch.cat(torch.chunk(torch.gather(input, dim, index), chunks=index.shape[0], dim=dim), dim=0)

#    @add_start_docstrings_to_model_forward(REALM_FOR_OPEN_QA_DOCSTRING.format("1, sequence_length"))
#    @replace_return_docstrings(output_type=M3ForOpenQAOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids,
        input_projected_score,
        kb_answers,
        is_train,
        labels,
        return_dict=None,
    ):

        # embedder
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # CPU computation starts.
        # [batch_size, block_emb_size]
        batch_scores = torch.einsum("BD,QD->QB", self.block_emb, input_projected_score.to(self.block_emb.device))

        # [batch_size, searcher_beam_size] - 각 sample과 score가 높은 topk knowledge의 idx를 반환
        #retrieved_block_scores, retrieved_block_ids = torch.topk(batch_scores, k=self.searcher_beam_size, dim=-1)
        _, retrieved_block_ids = torch.topk(batch_scores, k=self.searcher_beam_size, dim=-1)

        # [batch_size, searcher_beam_size]
        retrieved_block_ids = retrieved_block_ids.squeeze(-1)
        # [batch_size, searcher_beam_size, projection_size]
        retrieved_block_emb = torch.cat([torch.index_select(self.block_emb, dim=0, index=i).unsqueeze(0) for i in retrieved_block_ids])
        # CPU computation ends.
        
        # [batch_size, searcher_beam_size] => byte text
        retrieved_blocks = np.take(self.block_records, indices=retrieved_block_ids.cpu(), axis=0)
        # concat_input 생성
        text = []; text_pair = []; topk_labels = []
        if is_train:
            for iidx in range(input_ids.size(0)):
                for kidx in range(retrieved_block_emb.size(1)):
                    text.append(self.tokenizer.decode(input_ids[iidx], skip_special_tokens=False).replace(" <pad>", "").replace("</s>", "").strip())
                    # XX top-k개 붙이는 부분 수정
                    if isinstance(retrieved_blocks[iidx], np.ndarray) is False:
                        retrieved_blocks[iidx]= np.array([retrieved_blocks[iidx]], dtype=object)
                    text_pair.append(retrieved_blocks[iidx][kidx].decode())
                    topk_labels.append(labels[iidx].tolist())
            labels = torch.tensor(topk_labels)
        else:
            for iidx in range(input_ids.size(0)):
                text.append(self.tokenizer.decode(input_ids[iidx], skip_special_tokens=False).replace(" <pad>", "").replace("</s>", "").strip())
                if isinstance(retrieved_blocks[iidx], np.ndarray) is False:
                    retrieved_blocks[iidx]= np.array([retrieved_blocks[iidx]], dtype=object)
                text_pair.append(retrieved_blocks[iidx][0].decode())
        # concat_input
        concat_inputs = self.tokenizer(text, text_pair, padding=True, truncation=True, \
                    return_special_tokens_mask=True, max_length=self.concat_max_length)
        concat_inputs_tensors = concat_inputs.convert_to_tensors("pt")
        
        has_answers = []
        for ridx, retrieved_block_id in enumerate(retrieved_block_ids.tolist()):
            each_answers = []
            for retrieved_id in retrieved_block_id : 
                if kb_answers.tolist()[ridx] == retrieved_id : each_answers.append(True)
                else : each_answers.append(False)
            has_answers.append(each_answers)
        retrieved_logits = torch.einsum("BDC,BCK->BDK", retrieved_block_emb.to(self.block_emb.device), input_projected_score.unsqueeze(2))

        return M3RetrieverOutput(
                concat_inputs = concat_inputs_tensors,
                has_answers = has_answers,
                retrieved_logits = retrieved_logits,
                retrieved_block_ids = retrieved_block_ids,
                labels = labels
        )

@dataclass
class M3EmbedderOutput(ModelOutput):
    projected_score: torch.FloatTensor = None

class M3Embedder(M3PreTrainedModel):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.embedder = transformers.T5EncoderModel(self.config)
        if config.hidden_size != config.projected_size:
            self.cls = M3ScorerProjection(self.config)
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        answer_ids=None,
        return_dict=None,
        position_ids=None,
        tokenizer =None,
        labels=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        # embedder
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedder_outputs = self.embedder(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # [batch_size, hidden_size]
        pooler_output=embedder_outputs.last_hidden_state[:,0]

        # [batch_size, retriever_proj_size]
        if self.config.hidden_size != self.config.projected_size: projected_score = self.cls(pooler_output)
        else: projected_score = pooler_output
        
        if not return_dict:
            return (projected_score,) 
        else:
            return M3EmbedderOutput(
                projected_score=projected_score,
            )

class M3T5Encoder(M3PreTrainedModel):
    def __init__(self, config, tokenizer=None, block_records=None):
        super().__init__(config)

        self.tokenizer=tokenizer
        self.block_records=block_records
        self.embedder = M3Embedder(self.config)
        self.register_buffer(
                "block_emb",
                torch.zeros(()).new_empty(
                    size=(config.num_block_records, config.projected_size),
                    dtype=torch.float32,
                    device=torch.device(torch.cuda.current_device()),
                ),
            )
        self.retriever = M3Retriever(self.config, tokenizer=self.tokenizer, block_records=self.block_records, block_emb=self.block_emb, labels=None)
        self.reader_encoder = transformers.T5EncoderModel(self.config)
        self.post_init()

    def block_embedding_to(self, device):
        """Send `self.block_emb` to a specific device.

        Args:
            device (`str` or `torch.device`):
                The device to which `self.block_emb` will be sent.
        """

        self.block_emb = self.block_emb.to(device)

    def forward(
        self,
        input_ids,
        kb_answers,
        is_train,
        attention_mask=None,
        token_type_ids=None,
        answer_ids=None,
        return_dict=None,
        position_ids=None,
        tokenizer =None,
        labels=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # embedder
        embedder_outputs= self.embedder(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # [batch_size, d_model]
        embedder_projected_score = embedder_outputs.projected_score
        # retriever
        retriever_outputs = self.retriever(input_ids, embedder_projected_score, kb_answers, is_train, labels)
        if retriever_outputs.has_answers is not None: 
            has_answers = torch.tensor(retriever_outputs.has_answers, dtype=torch.bool, device=self.embedder.device)
        concat_inputs = retriever_outputs.concat_inputs.to(self.embedder.device)
        labels = retriever_outputs.labels
        
        # [batch_size, search_beam_size, 1]
        retriever_logits = retriever_outputs.retrieved_logits
        retrieved_block_ids = retriever_outputs.retrieved_block_ids
        
        # t5_encoder     
        reader_encoder_outputs = self.reader_encoder(
            input_ids=concat_inputs.input_ids, # [0 : self.config.reader_beam_size],
            attention_mask=concat_inputs.attention_mask, #[0 : self.config.reader_beam_size],
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return M3T5EncoderModelOutput(
                last_hidden_state = reader_encoder_outputs.last_hidden_state,
                hidden_states =reader_encoder_outputs.hidden_states,
                attentions=reader_encoder_outputs.attentions,
                retriever_logits = retriever_logits,
                has_answers= has_answers,
                retrieved_block_ids = retrieved_block_ids,
                labels =labels,
        )

@dataclass
class M3T5EncoderModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    retriever_logits: torch.FloatTensor = None
    has_answers : Optional[bool] = None
    retrieved_block_ids: torch.Tensor = None
    labels: torch.FloatTensor = None

@dataclass
class M3T5GenerateOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    retriever_loss: Optional[torch.FloatTensor] = None
    reader_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    retrieved_block_ids : torch.Tensor = None
    
class M3T5Generate(M3PreTrainedModel):
    def __init__(self, config, tokenizer=None, block_records=None):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        self.m3_t5_encoder = M3T5Encoder(config, tokenizer=self.tokenizer, block_records=block_records)
        self.m3_t5_decoder = transformers.T5ForConditionalGeneration(config).get_decoder()
        self.lm_head= transformers.T5ForConditionalGeneration(config).get_output_embeddings()

    def forward(self,
                input_ids=None,
                labels=None,
                kb_answers = None,
                is_train=None,
                decoder_input_ids=None,
                decoder_inputs_embeds=None,
                encoder_outputs=None,
                past_key_values=None,
                attention_mask=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                pad_id=0,
                loss_func=torch.nn.CrossEntropyLoss(ignore_index=-100, reduce=False),
                predict=False,
                **kwargs
                ):
            if encoder_outputs is None:
                encoder_outputs = self.m3_t5_encoder(
                    input_ids=input_ids,
                    kb_answers = kb_answers,
                    is_train = is_train,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                encoder_outputs = M3EncoderModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                    decoder_input_ids=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
                )
            labels = encoder_outputs.labels.to(self.device)
            if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = self.m3_t5_decoder._shift_right(labels)

            hidden_states = encoder_outputs.last_hidden_state 
            if predict == False :
                has_answers = encoder_outputs.has_answers # tensor([False, False, False] 
                relevance_score = encoder_outputs.retriever_logits # [retriever_beam_size, reader_beam_size, 1]
            attention_mask = None

            decoder_outputs = self.m3_t5_decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = decoder_outputs[0]
            sequence_output = sequence_output * (self.config.d_model ** -0.5)
            lm_logits = self.lm_head(sequence_output)

            loss=None
            retriever_loss = None
            reader_loss = None
            if not predict:
                lm_logits = lm_logits.permute(0,2,1)
                reader_ce_loss= loss_func(lm_logits, labels)
                reader_ce_loss= reader_ce_loss.mean(dim=1)
                reader_ce_loss = reader_ce_loss.reshape(relevance_score.size(0), relevance_score.size(1))
                reader_ce_loss_max, _ = torch.max(reader_ce_loss, dim=1)
                margin_reader_ce_loss = reader_ce_loss - reader_ce_loss_max.unsqueeze(-1)
                margin_reader_ce_loss *=-1

                def marginal_log_loss(logits, is_correct):
                    """Loss based on the negative marginal log-likelihood."""
                    def mask_to_score(mask, dtype=torch.float32):
                        return (1.0 - mask.type(dtype)) * -10 #torch.finfo(dtype).min
                    # []
                    log_numerator = torch.logsumexp(logits.squeeze(-1) + mask_to_score(is_correct), dim=-1)
                    log_denominator = torch.logsumexp(logits.squeeze(-1), dim=-1)
                    return log_denominator - log_numerator

                if has_answers is not None:
                    retriever_correct = has_answers
                    any_retriever_correct = torch.any(retriever_correct)

                    retriever_loss = marginal_log_loss(relevance_score, retriever_correct)
                    retriever_loss = torch.sum(retriever_loss)/retriever_loss.size(0)

                    reader_correct = has_answers
                    any_reader_correct = torch.any(reader_correct)
                    reader_loss = marginal_log_loss(margin_reader_ce_loss.unsqueeze(-1), reader_correct)
                    reader_loss = torch.sum(reader_loss)/reader_loss.size(0)
                loss = (retriever_loss + reader_loss).mean()

            return M3T5GenerateOutput(
                loss=loss,
                retriever_loss=retriever_loss,
                reader_loss=reader_loss,
                logits=lm_logits,
                labels = labels,
                retrieved_block_ids=encoder_outputs.retrieved_block_ids,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
                )

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        kb_answers=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        labels=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=0,
        eos_token_id=1,
        length_penalty=None,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None,
        num_return_sequences=None,
        max_time=None,
        decoder_start_token_id=None,
        use_cache=None,
        num_beam_groups=None,
        diversity_penalty=None,
        prefix_allowed_tokens_fn=None,
        output_attentions=None,
        output_hidden_states=None,
        output_scores=None,
        return_dict_in_generate=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        remove_invalid_values=None,
        attention_mask=None,
        **model_kwargs,
        ):

        if max_length==None: max_length = self.config.max_length

        decoder_input_ids = torch.tensor([[pad_token_id]]*input_ids.shape[0]).to(self.device)
        encoder_outputs = None
        for i in range(max_length):
            model_output = self.forward(input_ids=input_ids, decoder_input_ids=decoder_input_ids, \
                    encoder_outputs=encoder_outputs, kb_answers=kb_answers, predict=True, \
                    attention_mask=attention_mask, labels=labels)

            logits = model_output.logits.detach()
            predict = torch.argmax(logits, dim=-1)[:,-1:]

            decoder_input_ids = torch.cat((decoder_input_ids, predict),1)

        predict = decoder_input_ids
        retrieved_block_ids = model_output.retrieved_block_ids
        return (predict, retrieved_block_ids)

@dataclass
class M3T5GenerateModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    retrieved_block_ids: torch.Tensor = None
