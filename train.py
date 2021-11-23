import torch

import opts
from dataloader import DataLoader
from utils.logger import *
from utils.load_save import *
from models import setup
import misc.utils as utils
from torch import nn
from eval_utils import eval_split


def train(opt):
    opt.use_att = True

    loader = DataLoader(opt)

    infos = load_info(opt)
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)

    if opt.load_best_score == 1:
        best_score = infos.get('best_score', None)

    classifier = setup(opt).train().cuda()
    crit = utils.ClassifyCriterion().cuda()
    optimizer = utils.build_optimizer(classifier.parameters(), opt)
    optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    models = {'classifier': classifier}
    optimizers = {'classifier': optimizer}
    save_nets_structure(models, opt)
    load_checkpoint(models, optimizers, opt)
    print('opt', opt)

    # crit = nn.DataParallel(crit)
    # classifier = nn.DataParallel(classifier)

    epoch_done = True

    while True:
        if epoch_done:
            epoch_done = False

        # 1. fetch a batch of data from train split
        data = loader.get_batch('train')
        proposals = [torch.from_numpy(p).cuda() for p in data['proposals']]
        labels = torch.from_numpy(data['labels']).cuda()
        roi_features = torch.from_numpy(data['roi_features']).cuda()
        masks = [torch.from_numpy(_).cuda() for _ in data['masks']]
        pairs = data['pairs']

        # 2. Forward models and compute loss
        torch.cuda.synchronize()
        outputs = classifier(proposals, pairs, roi_features, labels)
        loss = crit(outputs, masks)

        # 3. Update models
        # loss = loss.mean()
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        # Update the iteration and epoch
        iteration += 1
        # Write the training loss summary
        if (iteration % opt.log_loss_every == 0):
            # logging log
            logger.info("{} ({}), loss: {:.3f}".format(iteration, epoch, loss.item()))
            tb.add_values('loss', {'train': loss.item()}, iteration)

        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True

        # Make evaluation and save checkpoint
        if (opt.save_checkpoint_every > 0 and iteration % opt.save_checkpoint_every == 0) or (
                opt.save_checkpoint_every == -1 and epoch_done):
            print('starting validation...')

            # eval models
            eval_kwargs = {'split': 'val'}
            eval_kwargs.update(vars(opt))

            val_loss = eval_split(classifier, loader, crit, eval_kwargs)
            print('the loss of validation is {:.2}'.format(val_loss))
            # log val results
            tb.add_values('loss', {'val': val_loss}, epoch)
            tb.add_values('lr', {'lr': utils.get_lr(optimizer)}, epoch)
            # Save models if is improving on validation result
            cur_score = val_loss

            optimizer.scheduler_step(cur_score)

            best_flag = False
            if best_score is None or cur_score < best_score:
                best_score = cur_score
                best_flag = True

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            infos['best_score'] = best_score
            infos['opt'] = opt
            infos['val_result_history'] = val_result_history

            save_checkpoint(models, optimizers,
                            infos, best_flag, opt)

            # Stop if reaching max epochs
            if epoch > opt.max_epochs and opt.max_epochs != -1:
                break


if __name__ == '__main__':
    opt = opts.parse_opt()
    logger = define_logger(opt)
    tb = MyTensorboard(opt)
    train(opt)
