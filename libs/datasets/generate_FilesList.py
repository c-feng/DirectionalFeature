import os
import numpy as np
import json

def read_json(fpath):
    with open(fpath,'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    with open(fpath,'w') as f:
        json.dump(obj,f,indent=4)

def generate_fileslist():
    root = "/home/ffbian/chencheng/XieheCardiac/npydata/"

    cars = os.listdir(root)
    cars.sort()

    fileslist = []  # [(path, index_time), (), ...]
    # 疾病类型
    for car in cars:
        persons = os.listdir(os.path.join(root, car))
        persons.sort()

        # 病人个体
        for person in persons:
            sliceds = os.listdir(os.path.join(root, car, person, "imgs"))
            sliceds.sort()

            # 切片位置
            for sliced in sliceds:
                file_p = os.path.join(root, car, person, "imgs", sliced)
                npy = np.load(file_p)
                time_n = npy.shape[-1]  # 25, 20, 11, 50

                # 时序
                for i in range(time_n):
                    fileslist.append((file_p, i))

    out_dir = os.path.dirname(os.path.abspath(__file__))
    # write_json(fileslist, os.path.join(out_dir, "DataList.json"))
    print(len(fileslist))

def generate_train_test_list():
    json_file = "/home/fcheng/Cardia/DataList.json"
    fileslist = read_json(json_file)

    nums = len(fileslist)
    train_ind = set(np.random.choice(nums, size=int(np.ceil(0.8*nums)), replace=False))
    test_ind = set(np.arange(nums)) - train_ind

    test_ind = list(test_ind)
    test_ind.sort()

    train_list = [fileslist[fl] for fl in train_ind]
    test_list = [fileslist[fl] for fl in test_ind]
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    write_json(train_list, os.path.join(out_dir, "train.json"))
    write_json(test_list, os.path.join(out_dir, "test.json"))

def generate_N_list(N=50000):
    json_file = "/home/fcheng/Cardia/source_code/libs/datasets/DataList.json"
    fileslist = read_json(json_file)

    nums = len(fileslist)
    train_ind = set(np.random.choice(nums, size=N, replace=False))
    test_ind = set(np.arange(nums)) - train_ind

    test_ind = list(test_ind)
    test_ind.sort()

    train_list = [fileslist[fl] for fl in train_ind]
    test_list = [fileslist[fl] for fl in test_ind]
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    write_json(train_list, os.path.join(out_dir, "train_{}.json".format(N)))
    write_json(test_list, os.path.join(out_dir, "test_{}.json".format(N)))

def gene_uniform_List(ratio=0.8, N=None):
    root = "/home/ffbian/chencheng/XieheCardiac/2DUNet/UNet/libs/datasets/differentkind/"
    cars = os.listdir(root)
    cars = [c for c in cars if "test" not in c]
    cars.sort()

    train_List = []
    test_List = []

    total_num = 0
    for json_f in cars:
        json_list = read_json(os.path.join(root, json_f))
        total_num += len(json_list)

    if N is not None:
        ratio = N / total_num
    
    train_num = 0
    for json_f in cars:
        json_list = read_json(os.path.join(root, json_f))
        train_num += np.ceil(ratio*len(json_list))
        print(json_f, len(json_list), np.ceil(ratio*len(json_list)))
        
        ta_ind = set(np.random.choice(len(json_list), size=int(np.ceil(ratio*len(json_list))), replace=False))
        te_ind = set(np.arange(len(json_list))) - ta_ind

        ta_ind = list(ta_ind)
        ta_ind.sort()
        te_ind = list(te_ind)
        te_ind.sort()
        
        train_List += [json_list[i] for i in ta_ind]
        test_List += [json_list[i] for i in te_ind]

    print(total_num, train_num)
    print(len(train_List), len(test_List))
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    write_json(train_List, os.path.join(out_dir, "train_{}.json".format(N)))
    write_json(test_List, os.path.join(out_dir, "test_{}.json".format(N)))

if __name__ == "__main__":
    # generate_fileslist()
    # generate_train_test_list()
    # generate_N_list(N=30000)
    gene_uniform_List(N=30000)