import os
import nibabel as nib
import numpy as np
import json

'''
DCM,HCM,MINF,NOR,RV
'''

def load_nii(nii_path):
    data = nib.load(nii_path)
    img = data.get_data()
    affine = data.affine
    header = data.header
    return img ,affine,header


def read_Infocfg(cfg_path):
    patient_info = {}




    with open (cfg_path) as f_in:
        for line in f_in:
            l = line.rstrip().split(":")
            #l is the list of the patient_info
            print(len(l))
            patient_info[l[0]] = l[1]
        # print(patient_info)
        '''
        ['ED', ' 1']
        ['ES', ' 12']
        ['Group', ' DCM']
        ['Height', ' 184.0']
        ['NbFrame', ' 30']
        ['Weight', ' 95.0']
        {'ED': ' 1', 'ES': ' 12', 'Group': ' DCM', 'Height': ' 184.0', 'NbFrame': ' 30', 'Weight': ' 95.0'}
        '''
    return patient_info

def read_json(fpath):
    with open(fpath,'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    with open(fpath,'w') as f:
        json.dump(obj,f,indent=4)

def write_json_append(obj, fpath):
    with open(fpath,'a') as f:
        json.dump(obj,f,indent=4)

def gen_alldatalist(path):
    filelist = []
    for dir in os.listdir(path):

        for file in os.listdir(os.path.join(root_path,dir)):
            filelist.append(os.path.join(root_path,dir,file))
    #the length of the filelist is 1902
    # print(len(filelist))
    filelist.sort()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    #/home/ffbian/chencheng/XieheCardiac/2DUNet/UNet/libs/datasets
    write_json(filelist, os.path.join(out_dir, "./acdcjson/ACDCDataList.json"))

def gen_every_kind_datalist(kind_path):
    filelist = []

    for file in os.listdir(kind_path):
        filelist.append(os.path.join(kind_path,file))

    filelist.sort()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    #/home/ffbian/chencheng/XieheCardiac/2DUNet/UNet/libs/datasets
    write_json(filelist, os.path.join(out_dir, "./acdcjson/{}DataList.json".format(kind_path[-2:])))

def generate_train_test_list(json_file_path):
    # json_file = "/home/fcheng/Cardia/DataList.json"
    # json_file = "/home/ffbian/chencheng/XieheCardiac/2DUNet/UNet/libs/datasets/ACDCDataList.json"
    fileslist = read_json(json_file_path)

    nums = len(fileslist)
    train_ind = set(np.random.choice(nums, size=int(np.ceil(0.8 * nums)), replace=False))
    test_ind = set(np.arange(nums)) - train_ind

    test_ind = list(test_ind)
    test_ind.sort()

    train_list = [fileslist[fl] for fl in train_ind]
    test_list = [fileslist[fl] for fl in test_ind]

    out_dir = os.path.dirname(os.path.abspath(__file__))
    print(out_dir)
    write_json(train_list, os.path.join(out_dir, "./acdcjson/RVtrain.json"))
    write_json(test_list, os.path.join(out_dir, "./acdcjson/RVtest.json"))
    write_json_append(train_list, os.path.join(out_dir, "./acdcjson/train.json"))
    write_json_append(test_list, os.path.join(out_dir, "./acdcjson/test.json"))
    # write_json(test_list, "/home/ffbian/chencheng/XieheCardiac/2DUNet/UNet/libs/datasets/differentkind/kuodatest.json")


if __name__ == "__main__":
    # read_Infocfg(os.path.join(root_path,"./DCM/patient001/Info.cfg"))
    '''
    a,b,c = load_nii(os.path.join(root_path,"./patient001/patient001_frame01.nii.gz"))
    print(np.unique(a))

    print(c)
    '''
    root_path = "/home/ffbian/chencheng/MICCAIACDC2017/processed_acdc_dataset/hdf5_files/Bykind"
    kind_path = "/home/ffbian/chencheng/MICCAIACDC2017/processed_acdc_dataset/hdf5_files/Bykind/RV"

    # gen_alldatalist(root_path)
    json_file_path = "/home/ffbian/chencheng/MICCAIACDC2017/mycode/libs/dataset/acdcjson/RVDataList.json"

    # generate_train_test_list(json_file_path)
    # gen_every_kind_datalist(kind_path)
    result = read_json(json_file_path)
    print(result)