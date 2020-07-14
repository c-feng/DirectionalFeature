import os
import argparse

def test_df(log_dir, epoch_i=0, best_model=False):
    OUTPUT_DIR = os.path.join(log_dir, "eval")
    if best_model:
        MODEL_PATH = os.path.join(log_dir, "ckpt", "model_best.pth")
    else:
        MODEL_PATH = os.path.join(log_dir, "ckpt", "checkpoint_epoch_{}.pth".format(epoch_i))

    commands = "python tools/test_df.py --used_df U_NetDF --selfeat --mgpus 6 --model_path1 {} \
                    --output_dir {} --log_file ../log_evaluation_vis.txt --vis".format(MODEL_PATH, OUTPUT_DIR)
    os.system(commands)

def train():
    os.system("python -m torch.distributed.launch --nproc_per_node 2 --master_port $RANDOM tools/train.py --batch_size 24 --mgpus 2,3 --output_dir logs/acdc_logs/log_temp --train_with_eval")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--scrip", type=str, default=None, help="which scrips to running")
    args = parser.parse_args()

    if args.scrip == "train":
        train()
    elif args.scrip == "test_df":
        test_df("logs/acdc_logs/logs_256_supcat_auxseg_thresh0.1/", best_model=False, epoch_i=118)