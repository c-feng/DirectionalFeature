import os
import sys
import numpy as np
import nibabel as nib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import _init_paths
from libs.network import U_Net, U_NetDF
from utils.image_list import to_image_list
import libs.datasets.augment as standard_augment
import libs.datasets.joint_augment as joint_augment

def progress_bar(curr_idx, max_idx, time_step, repeat_elem = "_"):
    max_equals = 55
    step_ms = int(time_step*1000)
    num_equals = int(curr_idx*max_equals/float(max_idx))
    len_reverse =len('Step:%d ms| %d/%d ['%(step_ms, curr_idx, max_idx)) + num_equals
    sys.stdout.write("Step:%d ms|%d/%d [%s]" %(step_ms, curr_idx, max_idx, " " * max_equals,))
    sys.stdout.flush()
    sys.stdout.write("/b" * (max_equals+1))
    sys.stdout.write(repeat_elem * num_equals)
    sys.stdout.write("/b"*len_reverse)
    sys.stdout.flush()
    if curr_idx == max_idx:
        print('/n')

def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header

def save_nii(vol, affine, hdr, path, prefix, suffix):
    vol = nib.Nifti1Image(vol, affine, hdr)
    vol.set_data_dtype(np.uint8)
    nib.save(vol, os.path.join(path, prefix+'_'+suffix + ".nii.gz"))


def get_person_names(root_path=None):
    persons_name = os.listdir(root_path)
    persons_name = [pn for pn in persons_name if "patient" in pn]
    persons_name.sort()
    return persons_name

def get_patient_data(patient, root_path):
    patient_data = {}
    infocfg_p = os.path.join(root_path, patient, "Info.cfg")

    with open(infocfg_p) as f_in:
        for line in f_in:
            l = line.rstrip().split(": ")
            patient_data[l[0]] = l[1]

    ed_path = os.path.join(root_path, patient, "%s_frame%02d.nii.gz" % (patient, int(patient_data['ED'])))
    es_path = os.path.join(root_path, patient, "%s_frame%02d.nii.gz" % (patient, int(patient_data['ES'])))
    img_4d_path = os.path.join(root_path, patient, "{}_4d.nii.gz".format(patient))
    # ed_gt_path = os.path.join(root_path, patient, "%s_frame%02d_gt.nii.gz" % (patient, int(patient_data['ED'])))
    # es_gt_path = os.path.join(root_path, patient, "%s_frame%02d_gt.nii.gz" % (patient, int(patient_data['ES'])))

    ed, affine, hdr = load_nii(ed_path)
    patient_data['ED_VOL'] = np.swapaxes(ed, 0, -1)
    patient_data['3D_affine'] = affine
    patient_data['3D_hdr'] = hdr

    es, _, _ = load_nii(es_path)  # (w, h, slices)
    patient_data['ES_VOL'] = np.swapaxes(es, 0, -1)

    img_4d, affine_4d, hdr_4d = load_nii(img_4d_path)  # (w, h, slices, times)
    patient_data['4D'] = np.swapaxes(img_4d, 0, 1)
    patient_data['4D_affine'] = affine_4d
    patient_data['4D_hdr'] = hdr_4d

    patient_data['size'] = img_4d.shape[:2][::-1]
    patient_data['pid'] = patient

    # ed_gt = load_nii(ed_gt_path)
    # patient_data['ED_GT'] = np.swapaxes(ed_gt, 0, 1)

    # es_gt = load_nii(es_gt_path)
    # patient_data['ES_GT'] = np.swapaxes(es_gt, 0, 1)
    return patient_data

def test_it(model, data, device="cuda", used_df=True):
    model.eval()
    imgs = data

    imgs = imgs.to(device)
    # gts = gts.to(device)

    net_out = model(imgs)
    if used_df:
        preds_out = net_out[0]
        preds_df = net_out[1]
    else:
        preds_out = net_out[0]
        preds_df = None
    preds_out = nn.functional.softmax(preds_out, dim=1)
    _, preds = torch.max(preds_out, 1)
    preds = preds.unsqueeze(1)  # (N, 1, *)

    return preds, preds_df

def transform(imgs):
    mean = 63.19523533061758
    std = 70.74166957523165
    trans = standard_augment.Compose([standard_augment.To_PIL_Image(),
                                    # joint_augment.RandomAffine(0,translate=(0.125, 0.125)),
                                    # joint_augment.RandomRotate((-180,180)),
                                    # joint_augment.FixResize(224),
                                    standard_augment.to_Tensor(),
                                    standard_augment.normalize([mean], [std]),
                                      ])
    return trans(imgs)

def test_voxel(model, imgs, used_df, multi_batches=False, resize=None):
    """ imgs: (slices, H, W)
        preds: (slices, 1, H, W)
    """
    imgs = imgs[..., None].astype(np.float32)
    B, _, _, C = imgs.shape

    if multi_batches:
        data, origin_shape = to_image_list(imgs, size_divisible=32, return_size=True)
        preds, _ = test_it(model, data)
        
        # for j in range(imgs.shape[0]):
        #     preds[j, ...] = pred.cpu().numpy()[j, :, :origin_shape[j][0], :origin_shape[j][1]]
    else:
        preds = torch.zeros(B, C, resize[0], resize[1])
        for j, pt in enumerate(imgs):
            data = [transform(pt)]
            data, origin_shape = to_image_list(data, size_divisible=32, return_size=True)
            pred, _ = test_it(model, data, used_df=used_df)
            preds[j, ...] = pred[0, 0, :origin_shape[0][0], :origin_shape[0][1]]

    if resize is not None:
        # preds = F.interpolate(preds, size=resize, mode='nearest')
        preds = preds[..., :resize[0], :resize[1]]

    return preds.cpu().numpy()[:, 0, ...]

def create_model(model_name, selfeat):
    if model_name == 'U_NetDF':
        model = U_NetDF(selfeat=selfeat, auxseg=True)
    elif model_name == 'U_Net':
        model = U_Net()
    # elif model_name == 'Resnet18_DfUnet':
    #     model = Resnet18_DfUnet()
    # elif model_name == 'DenseNet':
    #     model = DenseNet()
    # elif model_name == 'DenseNet_DF':
    #     model = DenseNet_DF(selfeat=selfeat)

    return model

def test(mgpus, model_name, model_path, used_df, selfeat, log_path):

    model = create_model(model_name, selfeat=selfeat)
    if mgpus is not None and len(mgpus) > 2:
        model = nn.DataParallel(model)
    model.cuda()

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])

    root_path = "MICCAIACDC2017/ACDC_DataSet/testing/testing/"
    root_path = "/root/ACDC_DataSet/testing/testing/"
    persons_name = get_person_names(root_path)
    for j, pn in enumerate(persons_name):
        s_time = time.time()
        patient_data = get_patient_data(pn, root_path)

        # (slices, h, w)
        es_pred = test_voxel(model, patient_data['ES_VOL'], used_df=used_df, resize=patient_data['size'])
        ed_pred = test_voxel(model, patient_data['ED_VOL'], used_df=used_df, resize=patient_data['size'])
        es_pred = np.transpose(es_pred, (2, 1, 0))
        ed_pred = np.transpose(ed_pred, (2, 1, 0))

        img_4D = patient_data['4D']
        h, w, s, t = img_4D.shape
        pred_4D = np.zeros((w, h, s, t))
        for i in range(img_4D.shape[-1]):
            pred = test_voxel(model, np.transpose(img_4D[...,i], (2, 0, 1)), used_df=used_df, resize=patient_data['size'])
            pred_4D[..., i] = np.transpose(pred, (2, 1, 0))
        
        save_path = os.path.join(log_path, "all_predictions")
        os.makedirs(save_path, exist_ok=True)
        CheckSizeAndSaveVolume(pred_4D, patient_data, save_path)
        progress_bar(j%(len(persons_name)+1), len(persons_name),time.time()-s_time)


def CheckSizeAndSaveVolume(seg_4D, patient_data, save_path):
    """
    TODO:
    """ 
    prefix = patient_data['pid']
    suffix = '4D'

    # save_nii(seg_4D, patient_data['4D_affine'], patient_data['4D_hdr'], save_path, prefix, suffix)
    suffix = 'ED'
    ED_phase_n = int(patient_data['ED'])
    ED_pred = seg_4D[:,:,:,ED_phase_n]
    save_nii(ED_pred.astype(np.uint8), patient_data['3D_affine'], patient_data['3D_hdr'], save_path, prefix, suffix)

    suffix = 'ES'
    ES_phase_n = int(patient_data['ES'])
    ES_pred = seg_4D[:,:,:,ES_phase_n]
    save_nii(ES_pred.astype(np.uint8), patient_data['3D_affine'], patient_data['3D_hdr'], save_path, prefix, suffix)

    # ED_GT = patient_data.get('ED_GT', None)
    results = []
    return results



if __name__ == "__main__":
    # get_person_names()
    import argparse
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument('--mgpus', type=str, default='0', required=False, help='whether to use multiple gpu')
    parser.add_argument('--used_df', type=str, default='True', help='whether to use df')
    parser.add_argument('--model', type=str, default='', help='whether to use df')
    parser.add_argument('--selfeat', type=bool, default=True, help='whether to use feature select')
    parser.add_argument('--model_path', type=str, default=None, help='whether to train with evaluation')
    parser.add_argument('--output_dir', type=str, default="logs/acdc_logs/logs_supcat_auxseg/predtions", required=False, help='specify an output directory if needed')
    parser.add_argument('--log_file', type=str, default="../log_predtion.txt", help="the file to write logging")

    args = parser.parse_args()
    if args.mgpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.mgpus

    model_path = "logs/acdc_logs/logs_supcat_auxseg/ckpt/checkpoint_epoch_70.pth"

    test(args.mgpus, "U_NetDF", model_path, args.used_df, args.selfeat, args.output_dir)
