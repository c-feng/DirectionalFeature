import json, h5py
# acdc_test_json = "/home/ffbian/chencheng/MICCAIACDC2017/mycode/libs/dataset/acdcjson/test.json"
# acdc_train_json = "/home/ffbian/chencheng/MICCAIACDC2017/mycode/libs/dataset/acdcjson/train.json"
acdc_test_json = "/home/ffbian/chencheng/MICCAIACDC2017/mycode/libs/dataset/acdcjson/Dense_TestList.json"
acdc_train_json = "/root/chengfeng/Cardiac/source_code/libs/datasets/jsonLists/acdcList/Dense_TrainList.json"


def func(path):
    with open(path, 'r') as f:
        test_list = json.load(f)
    print(len(test_list))

    name_list = set()
    for tl in test_list:
        a0, a1 = tl.split('/')[-2:]
        a1 = "_".join(a1.split('_')[:2])
        name_list.add(a0+'-'+a1)

    name_list = list(name_list)
    name_list.sort()
    return name_list

name_list = func(acdc_train_json)
# name_list = func(acdc_test_json)
# name_list = name_list_train + name_list_test
name_list.sort()

# outfile = "./AcdcPersonCarname.txt"
# with open(outfile, "w") as f:
#     # for nl in name_list:
#     f.writelines('\n'.join(name_list))

outfile = "./AcdcDenseTrainPersonCarname.json"
with open(outfile, "w") as f:
    json.dump(name_list, f)

