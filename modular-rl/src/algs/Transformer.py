import math
import copy

import torch
from torch import nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    """

    def __init__(self, encoder_layer, num_layers, norm=None, used_by=None, args=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.used_by = used_by
        self.args = args

    def forward(self, src, synergy, mask=None, src_key_padding_mask=None, env_name = None):
        r"""Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        if self.args.enable_synergy_obs or self.args.enable_synergy_act:
            n_agents = len(output)
            if self.num_layers == 2:
                intra_synergy = [0]
                inter_synergy = [1]
            elif self.num_layers == 3:
                intra_synergy = [0]
                inter_synergy = [1,2]
            else:
                raise NotImplementedError

            for i in intra_synergy:
                output = self.layers[i](output, src_mask=synergy.get_mask(env_name) if self.args.enable_synergy_obs else None,
                                    src_key_padding_mask=src_key_padding_mask)

            if self.args.enable_synergy_act: #and self.used_by == 'policy':
                w = synergy.get_merge_weight(env_name)
                output_size = output.shape
                output = torch.matmul(w, output.reshape(output_size[0],-1)).reshape(-1, output_size[1], output_size[2])

            for i in inter_synergy:
                output = self.layers[i](output, src_mask=None,
                                        src_key_padding_mask=src_key_padding_mask)

        else:
            for i in range(self.num_layers):
                output = self.layers[i](output, src_mask=None,
                                        src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            output = self.norm(output)

        return output

class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

class TransformerModel(nn.Module):
    def __init__(
        self,
        feature_size,
        output_size,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.5,
        condition_decoder=False,
        transformer_norm=False,
        used_by=None,
        args=None,
    ):
        """This model is built upon https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.args=args
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            nlayers,
            norm=nn.LayerNorm(ninp) if transformer_norm else None,
            used_by=used_by,
            args=args,
        )

        self.encoder = nn.Linear(feature_size, ninp)
        self.ninp = ninp
        self.condition_decoder =  condition_decoder
        if self.args.enable_synergy_act and used_by == 'policy':
            decoder_output_size = args.synergy_action_dim * 1 #* 64
        else:
            decoder_output_size = output_size
        decoder_input_size = ninp + feature_size if condition_decoder else ninp
        self.decoder = nn.Linear(
            decoder_input_size, decoder_output_size
        )
        D_AGG = 4
        self.act_weight = nn.Sequential(
            nn.Linear(ninp + args.d_model * D_AGG, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        self.init_weights()

        self.used_by = used_by
        print(self.args.enable_synergy_obs, self.args.enable_synergy_act)
        self.print_counter = 0

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, synergy, mode='train', env_name=None):
        n_agents, bs, dim = src.shape
        encoded = self.encoder(src) * math.sqrt(self.ninp)
        output1 = self.transformer_encoder(encoded, synergy, env_name=env_name) # [K,bs,d]

        if self.condition_decoder:
            if self.args.enable_synergy_act:
                w = synergy.get_merge_weight(env_name)
                bypass_src = (w@src.reshape(n_agents, -1)).reshape(-1,bs,dim)
            else:
                bypass_src = src
            output1 = torch.cat([output1, bypass_src], dim=2)

        output = self.decoder(output1) # [K,bs,d] ==> [K,bs,SD/1]

        if self.used_by == 'policy':
            if self.args.enable_synergy_act:
                w = synergy.get_action_weight(env_name).unsqueeze(0)  # [1,1/SD,N,K]
                embeds = synergy.get_embed(env_name).repeat(bs, 1, 1)  # [1,n,d] => [bs,n,d]
                state = encoded.permute(1, 0, 2)
                inpt = torch.cat([embeds, state], dim=-1)  # [bs, n, d]
                # inpt = encoded.permute(1, 0, 2)

                w1 = self.act_weight(inpt).permute(0, 2, 1).unsqueeze(-1)  # [1,1/SD,N,1]
                w = w * w1  # [bs, 1/SD, n, K]

                output = output.permute(1,2,0).unsqueeze(-1) # [bs,SD/1,K,1]

                output = (w @ output).mean(dim=1).permute(1, 0, 2)  # [bs,SD/1,N,1] => [bs,N,1] => [N,bs,1]
            output = output.tanh() # [n,bs,1]

        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])