import os
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import json

from utils.comm import get_world_size, get_rank
import utils.metrics as metrics
from utils.image_list import to_image_list
from tools.test_utils import personTo4Ddata, test_person
from libs.datasets import joint_augment as joint_augment
from libs.datasets import augment as standard_augment

def save_checkpoint(state, filename='checkpoint', is_best=False):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.join(os.path.dirname(filename), "model_best.pth"))

def checkpoint_state(model=None, optimizer=None, epoch=None, it=None, performance=0.):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'performance': performance}

def load_checkpoint(model=None, optimizer=None, filename="checkpoint", logger=None):
    if os.path.isfile(filename):
        if logger is not None:
            logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location="cpu")
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        performance = checkpoint.get('performance', 0.)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if logger is not None:
            logger.info("==> Done")
    else:
        raise FileNotFoundError
    
    return it, epoch, performance

class Trainer():
    def __init__(self, model, model_fn, criterion, optimizer, ckpt_dir, lr_scheduler, model_fn_eval,
                 tb_log, logger, eval_frequency=1, grad_norm_clip=1.0, cfg=None):
        self.model, self.model_fn, self.optimizer, self.model_fn_eval = model, model_fn, optimizer, model_fn_eval

        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.ckpt_dir = ckpt_dir
        self.tb_log = tb_log
        self.logger = logger
        self.eval_frequency = eval_frequency
        self.grad_norm_clip = grad_norm_clip
        self.cfg = cfg
        self.caches_4D = {}

    def _train_it(self, batch, epoch=0):
        self.model.train()

        self.optimizer.zero_grad()
        loss, tb_dict, disp_dict = self.model_fn(self.model, batch, self.criterion, perfermance=False, epoch=0)

        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item(), tb_dict, disp_dict
    
    def eval_epoch(self, d_loader):
        self.model.eval()

        eval_dict = {}
        total_loss = 0

        # eval one epoch
        if get_rank() == 0: print("evaluating...")
        sel_num = np.random.choice(len(d_loader), size=1)
        for i, data in enumerate(d_loader, 0):
            self.optimizer.zero_grad()
            vis = True if i == sel_num else False

            loss, tb_dict, disp_dict = self.model_fn_eval(self.model, data, self.criterion, perfermance=True, vis=vis)

            total_loss += loss.item()

            for k, v in tb_dict.items():
                if "vis" not in k:
                    eval_dict[k] = eval_dict.get(k, 0) + v
                else:
                    eval_dict[k] = v
            if get_rank() == 0: print("\r{}/{} {:.0%}\r".format(i, len(d_loader), i/len(d_loader)), end='')
        if get_rank() == 0: print()

        for k, v in tb_dict.items():
            if "vis" not in k:
                eval_dict[k] = eval_dict.get(k, 0) / (i + 1)
        
        return total_loss / (i+1), eval_dict, disp_dict

    def train(self, start_it, start_epoch, n_epochs, train_loader, test_loader=None,
              ckpt_save_interval=5, lr_scheduler_each_iter=False, best_res=0):
        eval_frequency = self.eval_frequency if self.eval_frequency else 1

        it = start_it
        for epoch in range(start_epoch, n_epochs):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)
            
            for cur_it, batch in enumerate(train_loader):
                cur_lr = self.lr_scheduler.get_lr()[0]

                loss, tb_dict, disp_dict = self._train_it(batch, epoch)
                it += 1

                # print infos
                if get_rank() == 0:
                    print("Epoch/train:{}({:.0%})/{}({:.0%})".format(epoch, epoch/n_epochs,
                                    cur_it, cur_it/len(train_loader)), end="")
                    for k, v in disp_dict.items():
                        print(", ", k+": {:.6}".format(v), end="")
                    print("")

                # tensorboard logs
                if self.tb_log is not None:
                    self.tb_log.add_scalar("train_loss", loss, it)
                    self.tb_log.add_scalar("learning_rate", cur_lr, it)
                    for key, val in tb_dict.items():
                        self.tb_log.add_scalar('train_'+key, val, it)

            # save trained model
            trained_epoch = epoch
            # if trained_epoch % ckpt_save_interval == 0:
            #     ckpt_name = os.path.join(self.ckpt_dir, "checkpoint_epoch_%d" % trained_epoch)
            #     save_checkpoint(checkpoint_state(self.model, self.optimizer, trained_epoch, it),
            #                     filename=ckpt_name)

            # eval one epoch
            if (epoch % eval_frequency) == 0 and (test_loader is not None):
                with torch.set_grad_enabled(False):
                    val_loss, eval_dict, disp_dict = self.eval_epoch(test_loader)
                    # mean_3D = self.metric_3D(self.model, self.cfg)

                if self.tb_log is not None:
                    for key, val in eval_dict.items():
                        if "vis" not in key:
                            self.tb_log.add_scalar("val_"+key, val, it)
                        else:
                            self.tb_log.add_images("df_gt", val[0], it, dataformats="NCHW")
                            self.tb_log.add_images("df_pred", val[2], it, dataformats="NCHW")
                            self.tb_log.add_images("df_magnitude", val[1], it, dataformats="NCHW")

                # save model and best model
                if get_rank() == 0:
                    # cal 3D dice
                    # if self.tb_log is not None:
                    #     for k, v in mean_3D.items():
                    #         self.tb_log.add_scalar("val_3D_"+k, v, it)

                    res = np.mean([eval_dict["LV_dice"], eval_dict["RV_dice"], eval_dict["MYO_dice"]])
                    # res = np.mean([mean_3D["LV_dice"], mean_3D["RV_dice"], mean_3D["MYO_dice"]])
                    self.logger.info("Epoch {} mean dice(2D/3D): {}/N".format(epoch, res))
                    if best_res != 0:
                        _, _, best_res = load_checkpoint(filename=os.path.join(self.ckpt_dir, "model_best.pth"))
                    is_best = res > best_res
                    best_res = max(res, best_res)

                    ckpt_name = os.path.join(self.ckpt_dir, "checkpoint_epoch_%d" % trained_epoch)
                    save_checkpoint(checkpoint_state(self.model, self.optimizer, trained_epoch, it, performance=res),
                                    filename=ckpt_name, is_best=is_best)

    def metric_3D(self, model, cfg):
        p_json = cfg.DATASET.TEST_PERSON_LIST
        datadir_4D = "/root/ACDC_DataSet/4dData"

        with open(p_json, "r") as f:
            persons = json.load(f)
        
        total_segMetrics = {"dice": [[], [], []],
                        "hausdorff": [[], [], []]}
        for i, p in enumerate(persons):
            # imgs, gts = personTo4Ddata(p, val_list)
            if p in self.caches_4D.keys():
                imgs, gts = self.caches_4D[p]
            else:
                imgs = np.load(os.path.join(datadir_4D, p.split('-')[1], '4d_data.npy'))
                gts = np.load(os.path.join(datadir_4D, p.split('-')[1], '4d_gt.npy'))
                self.caches_4D[p] = [imgs, gts]

            imgs, gts = imgs.astype(np.float32)[..., None, :], gts.astype(np.float32)[..., None,:]
            imgs, gts = joint_transform(imgs, gts, cfg)
            gts = [gt[:, 0, ...].numpy() for gt in gts]

            preds = test_person(model, imgs, multi_batches=True, used_df=cfg.DATASET.DF_USED)  # (times, slices, H, W)

            segMetrics = {"dice": [], "hausdorff": []}
            for j in range(len(preds)):
                segMetrics["dice"].append(metrics.dice3D(preds[j], gts[j], gts[j].shape))
                segMetrics["hausdorff"].append(metrics.hd_3D(preds[j], gts[j]))
            
            for k, v in segMetrics.items():
                segMetrics[k] = np.array(v).reshape((-1, 3))

            for k, v in total_segMetrics.items():
                for j in range(3):
                    total_segMetrics[k][j] += segMetrics[k][:, j].tolist()
            # person i is done
            if get_rank() == 0: print("\r{}/{} {:.0%}\r".format(i, len(persons), i/len(persons)), end='')
        if get_rank() == 0: print()

        mean = {}
        for k, v in total_segMetrics.items():
            mean.update({"LV_"+k: np.mean(v[1])})
            mean.update({"MYO_"+k: np.mean(v[2])})
            mean.update({"RV_"+k: np.mean(v[0])})
        return mean

def transform(imgs, cfg):
    trans = standard_augment.Compose([standard_augment.normalize([cfg.DATASET.MEAN], [cfg.DATASET.STD]),
                                      ])
    return trans(imgs)
def joint_transform(imgs, gts, cfg):
    trans = joint_augment.Compose([joint_augment.To_PIL_Image(),
                                #    joint_augment.RandomAffine(0,translate=(0.125, 0.125)),
                                #    joint_augment.RandomRotate((-180,180)),
                                   joint_augment.FixResize(256),
                                   joint_augment.To_Tensor()
                                   ])
    S, H, W, C, T = gts.shape
    trans_imgs = [None] * T
    trans_gts = [None] * T
    for i in range(T):
        trans_imgs[i], trans_gts[i] = [], []
        for j in range(S):
            t0, t1 = trans(imgs[j,...,i], gts[j,...,i])
            trans_imgs[i].append(transform(t0, cfg))
            trans_gts[i].append(t1)

    aligned_imgs = []
    aligned_gts = []
    for i in range(T):
        aligned_imgs.append(to_image_list(trans_imgs[i], size_divisible=32))
        aligned_gts.append(to_image_list(trans_gts[i], size_divisible=32))

    return aligned_imgs, aligned_gts

