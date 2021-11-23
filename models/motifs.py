# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from util import cat, obj_edge_vectors, center_x, to_onehot, sort_by_score, get_dropout_mask, nms_overlaps, \
    encode_box_info, layer_init, get_dataset_statistics


class LSTMContext(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    """

    def __init__(self, config, obj_names, in_channels):
        super(LSTMContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_names
        self.num_obj_classes = len(obj_names)

        # word embedding
        self.embed_dim = self.cfg.embed_dim
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.glove_dir, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        # self.pos_embed = nn.Sequential(*[
        #     nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
        #     nn.Linear(32, 128), nn.ReLU(inplace=True),
        # ])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.dropout_rate
        self.hidden_dim = self.cfg.hidden_dim
        self.nl_obj = self.cfg.obj_layer
        self.nl_edge = self.cfg.rel_layer
        assert self.nl_obj > 0 and self.nl_edge > 0

        # TODO Kaihua Tang
        # AlternatingHighwayLSTM is invalid for pytorch 1.0
        self.obj_ctx_rnn = torch.nn.LSTM(
            input_size=self.obj_dim + self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_obj,
            dropout=self.dropout_rate if self.nl_obj > 1 else 0,
            bidirectional=True)
        self.edge_ctx_rnn = torch.nn.LSTM(
            input_size=self.embed_dim + self.hidden_dim + self.obj_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_edge,
            dropout=self.dropout_rate if self.nl_edge > 1 else 0,
            bidirectional=True)
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.lin_obj_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.lin_edge_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def sort_rois(self, proposals):
        c_x = center_x(proposals)
        # leftright order
        scores = c_x / (c_x.max() + 1)
        return sort_by_score(proposals, scores)

    def obj_ctx(self, obj_feats, proposals):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :param box_priors: [num_obj, 4] boxes. We'll use this for NMS
        :param boxes_per_cls
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        perm, inv_perm, ls_transposed = self.sort_rois(proposals)
        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        encoder_rep = self.lin_obj_h(encoder_rep)  # map to hidden_dim

        # obj_preds = obj_labels
        # obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        encoder_rep = encoder_rep[inv_perm]

        return encoder_rep, perm, inv_perm, ls_transposed

    def edge_ctx(self, inp_feats, perm, inv_perm, ls_transposed):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]
        edge_reps = self.lin_edge_h(edge_reps)  # map to hidden_dim

        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def forward(self, x, proposals, labels):
        # labels will be used in DecoderRNN during training (for nms)
        obj_embed = self.obj_embed1(labels.long())
        obj_pre_rep = cat((x, obj_embed), -1)

        # object level contextual feature
        obj_ctx, perm, inv_perm, ls_transposed = self.obj_ctx(obj_pre_rep, proposals)
        # edge level contextual feature
        obj_embed2 = self.obj_embed2(labels.long())

        obj_rel_rep = cat((obj_embed2, x, obj_ctx), -1)

        edge_ctx = self.edge_ctx(obj_rel_rep, perm=perm, inv_perm=inv_perm, ls_transposed=ls_transposed)

        return edge_ctx


class MotifPredictor(nn.Module):
    def __init__(self, config):
        super(MotifPredictor, self).__init__()
        self.in_channels = config.in_channels
        self.num_obj_cls = config.obj_num_classes
        self.num_rel_cls = config.rel_num_classes

        assert self.in_channels is not None

        # load class dict
        obj_names = get_dataset_statistics()
        assert self.num_obj_cls == len(obj_names)
        # init contextual lstm encoding
        self.context_layer = LSTMContext(config, obj_names, self.in_channels)

        # post decoding
        self.hidden_dim = config.hidden_dim
        self.pooling_dim = config.pooling_dim
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)
        self.sigmoid = nn.Sigmoid()

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

    def forward(self, proposals, rel_pair_idxs, roi_features, labels):
        """
        Returns:
            rel_probs (list[Tensor])
        """
        # encode context infomation
        edge_ctx = self.context_layer(roi_features, proposals, labels)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [b.shape[0] for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        prod_reps = []
        for pair_idx, head_rep, tail_rep in zip(rel_pair_idxs, head_reps, tail_reps):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))

        prod_rep = cat(prod_reps, dim=0)
        prod_rep = self.post_cat(prod_rep)

        rel_dists = self.rel_compress(prod_rep)
        rel_prob = self.sigmoid(rel_dists).split(num_rels, dim=0)

        return rel_prob
