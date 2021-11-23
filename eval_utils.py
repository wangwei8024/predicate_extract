from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def eval_split(model, loader, crit, eval_kwargs={}):
    split = eval_kwargs.get('split', 'val')
    num_images = len(loader.split_ix[split])

    # Make sure in the evaluation mode
    model.eval()
    loader.reset_iterator(split)
    n = 0
    losses = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        proposals = [torch.from_numpy(p).cuda() for p in data['proposals']]
        labels = torch.from_numpy(data['labels']).cuda()
        roi_features = torch.from_numpy(data['roi_features']).cuda()
        masks = [torch.from_numpy(_).cuda() for _ in data['masks']]
        pairs = data['pairs']

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            outputs = model(proposals, pairs, roi_features, labels)
            loss = crit(outputs, masks)
            losses.append(loss.item())

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    model.train()
    return sum(losses) / len(losses)
