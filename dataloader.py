from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import random
import logging
import pickle
import torch
import torch.utils.data as data


class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split == 'train', self.opt.loader_num_workers,
                                                    self.opt)
        self.iterators[split] = 0

    def __init__(self, opt):
        self.opt = opt
        self.att_feat_size = opt.att_feat_size

        self.logger = logging.getLogger('__main__')
        self.batch_size = self.opt.batch_size

        # data dir
        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir
        self.input_box_dir = self.opt.input_box_dir
        self.sg_data_dir = self.opt.sg_data_dir
        self.rel_num_classes = self.opt.rel_num_classes

        self.info = json.load(open(self.opt.input_json))
        self.bag_info = json.load(open(self.opt.bag_info_path))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            else:  # restval
                self.split_ix['train'].append(ix)
        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        for split in self.split_ix.keys():
            self.logger.info('assigned %d images to split %s' % (len(self.split_ix[split]), split))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        for split in self.split_ix.keys():
            self.logger.info('assigned %d images to split %s' % (len(self.split_ix[split]), split))

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train', self.opt.loader_num_workers,
                                                        self.opt)
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    def get_batch(self, split, batch_size=None):
        self.split = split
        batch_size = batch_size or self.batch_size

        att_batch = []
        box_batch = []
        label_batch = []
        pair_batch = []
        mask_batch = []

        infos = []
        wrapped = False

        for i in range(batch_size):
            # fetch image
            tmp_att, tmp_box, tmp_label, temp_pair, tmp_mask, ix, tmp_wrapped = self._prefetch_process[self.split].get()

            att_batch.append(tmp_att)
            box_batch.append(tmp_box)
            label_batch.append(tmp_label)
            pair_batch.append(temp_pair)
            mask_batch.append(tmp_mask)

            if tmp_wrapped:
                wrapped = True

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            infos.append(info_dict)

        data = {}
        data['roi_features'] = np.concatenate(att_batch, axis=0)
        data['proposals'] = box_batch
        data['labels'] = np.concatenate(label_batch, axis=0)
        data['pairs'] = pair_batch
        data['masks'] = mask_batch

        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        image_id = str(self.info['images'][ix]['id'])
        att_feat = np.load(os.path.join(self.input_att_dir, '{}.npz'.format(image_id)))['feat']
        # Reshape to K x C
        att_feat = att_feat.reshape(-1, att_feat.shape[-1])
        box_feat = self.get_box_feat(image_id)
        # get obj labels
        label = \
            np.load(os.path.join(self.sg_data_dir, '{}.npy'.format(image_id)), allow_pickle=True, encoding='latin1')[
                ()]['obj_attr'][:,
            1]
        # pairs
        pairs = []
        att_len = att_feat.shape[0]
        for ii in range(att_len):
            for jj in range(att_len):
                pairs.append(np.array([ii, jj], dtype='int'))

        pair = np.stack(pairs, axis=0)
        # mask
        mask = np.zeros([att_len * att_len, self.rel_num_classes], dtype='float32')

        info = self.bag_info[image_id]
        for key, values in info.items():
            for val in values:
                mask[val[0] * att_len + val[1], int(key)] = 1

        return (att_feat, box_feat, label, pair, mask, ix)

    def get_box_feat(self, image_id):
        box_feat = np.load(os.path.join(self.input_box_dir, '{}.npy'.format(image_id)))
        return box_feat


def __len__(self):
    return len(self.split_ix['train']) + len(self.split_ix['val']) + len(self.split_ix['test'])


class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False, num_workers=4, opt=None):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.opt = opt
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle
        self.num_workers = num_workers

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=1,
                                                 sampler=SubsetSampler(self.dataloader.split_ix[self.split][
                                                                       self.dataloader.iterators[self.split]:]),
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers,  # 4 is usually enough
                                                 worker_init_fn=None,
                                                 collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[-1] == ix, "ix not equal"

        return tmp + [wrapped]
