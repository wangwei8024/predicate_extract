import torch
import os
from tqdm import tqdm
import six
import sys
from six.moves.urllib.request import urlretrieve
import zipfile
import array
import numpy as np
import json


def obj_edge_vectors(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim).cuda()
    vectors.normal_(0, 1)

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Try the longest word
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print("fail on {}".format(token))

    return vectors


def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)

    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        try:
            return torch.load(fname_pt, map_location=torch.device("cpu"))
        except Exception as e:
            print("Error loading the models from {}{}".format(fname_pt, str(e)))
            sys.exit(-1)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def center_x(proposals):
    boxes = cat([p for p in proposals], dim=0)
    boxes = boxes.view(-1, 4)
    c_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    return c_x.view(-1)


def to_onehot(vec, num_classes, fill=1000):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = fill

    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :return:
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(-fill)
    arange_inds = vec.new(vec.size(0)).long()
    torch.arange(0, vec.size(0), out=arange_inds)

    onehot_result.view(-1)[vec.long() + num_classes * arange_inds] = fill
    return onehot_result


def transpose_packed_sequence_inds(lengths):
    """
    Get a TxB indices from sorted lengths.
    Fetch new_inds, split by new_lens, padding to max(new_lens), and stack.
    Returns:
        new_inds (np.array) [sum(lengths), ]
        new_lens (list(np.array)): number of elements of each time step, descending
    """
    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer + 1)].copy())
        cum_add[:(length_pointer + 1)] += 1
        new_lens.append(length_pointer + 1)
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens


def sort_by_score(proposals, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_rois = [len(b) for b in proposals]
    num_im = len(num_rois)

    scores = scores.split(num_rois, dim=0)
    ordered_scores = []
    for i, (score, num_roi) in enumerate(zip(scores, num_rois)):
        ordered_scores.append(score + 2.0 * float(num_roi * 2 * num_im + i))
    ordered_scores = cat(ordered_scores, dim=0)
    _, perm = torch.sort(ordered_scores, 0, descending=True)

    num_rois = sorted(num_rois, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(num_rois)  # move it to TxB form
    inds = torch.LongTensor(inds).to(scores[0].device)
    ls_transposed = torch.LongTensor(ls_transposed)

    perm = perm[inds]  # (batch_num_box, )
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, ls_transposed


def get_dropout_mask(dropout_probability, tensor_shape, device):
    """
    once get, it is fixed all the time
    """
    binary_mask = (torch.rand(tensor_shape) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().to(device).div(1.0 - dropout_probability)
    return dropout_mask


def nms_overlaps(boxes):
    """ get overlaps for each channel"""
    assert boxes.dim() == 3
    N = boxes.size(0)
    nc = boxes.size(1)
    max_xy = torch.min(boxes[:, None, :, 2:].expand(N, N, nc, 2),
                       boxes[None, :, :, 2:].expand(N, N, nc, 2))

    min_xy = torch.max(boxes[:, None, :, :2].expand(N, N, nc, 2),
                       boxes[None, :, :, :2].expand(N, N, nc, 2))

    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)

    # n, n, 151
    inters = inter[:, :, :, 0] * inter[:, :, :, 1]
    boxes_flat = boxes.view(-1, 4)
    areas_flat = (boxes_flat[:, 2] - boxes_flat[:, 0] + 1.0) * (
            boxes_flat[:, 3] - boxes_flat[:, 1] + 1.0)
    areas = areas_flat.view(boxes.size(0), boxes.size(1))
    union = -inters + areas[None] + areas[:, None]
    return inters / union


def encode_box_info(proposals, sizes):
    """
    encode proposed box information (x1, y1, x2, y2) to
    (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """
    n_bag = proposals.size(0)
    att_len = proposals.size(1)
    boxes_info = []
    for ii, proposal in enumerate(proposals):
        boxes = proposal
        wid = sizes[ii][0]
        hei = sizes[ii][1]
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        w, h = wh.split([1, 1], dim=-1)
        x, y = xy.split([1, 1], dim=-1)
        x1, y1, x2, y2 = boxes.split([1, 1, 1, 1], dim=-1)
        assert wid * hei != 0
        info = torch.cat([w / wid, h / hei, x / wid, y / hei, x1 / wid, y1 / hei, x2 / wid, y2 / hei,
                          w * h / (wid * hei)], dim=-1).view(-1, 9)
        boxes_info.append(info)
    return torch.cat(boxes_info, dim=0).float()


def layer_init(layer, init_para=0.1, normal=False, xavier=True):
    xavier = False if normal == True else True
    if normal:
        torch.nn.init.normal_(layer.weight, mean=0, std=init_para)
        torch.nn.init.constant_(layer.bias, 0)
        return
    elif xavier:
        torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
        torch.nn.init.constant_(layer.bias, 0)
        return


def get_dataset_statistics():
    sg_dict = np.load('data/coco_pred_sg_rela.npy', allow_pickle=True)[()]
    sg_dict = sg_dict['i2w']

    obj_names = [sg_dict[ii] for ii in range(305)]
    return obj_names
