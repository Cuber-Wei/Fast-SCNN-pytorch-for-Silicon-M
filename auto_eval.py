"""
author: L0v3ch4n
Eval the weights file in path: weights-backup/ automaticly.
The results will be written into evalRes.txt.
"""

import os
import shutil

from eval import Evaluator
from train import parse_args

WEIGHT_PATH = "weights"
WEIGHT_BACK_PATH = "weights-backup"

def eval_model(epoch, args):
    # backup: copy the weights file to the train-weights folder
    shutil.copyfile(
        os.path.join(WEIGHT_BACK_PATH, f"fast_scnn_citys-{epoch}.pth"),
        os.path.join(
            WEIGHT_BACK_PATH, "train-weights", f"fast_scnn_citys-{epoch}.pth"
        ),
    )
    args.idx = epoch
    evaluator = Evaluator(args)
    print(f"[INFO] Current Epoch: {epoch}, Testing model: {args.model}")
    # replace the weights file in the WEIGHTS_PATH
    os.replace(
        os.path.join(WEIGHT_BACK_PATH, f"fast_scnn_citys-{epoch}.pth"),
        os.path.join(WEIGHT_PATH, f"fast_scnn_citys.pth"),
    )
    # eval the model
    evaluator.eval()
    del evaluator
    # rename the result dir
    os.rename("test_result", f"test_result_{epoch}")

def main():
    args = parse_args()
    if not os.path.exists(os.path.join(WEIGHT_BACK_PATH, "train-weights")):
        os.makedirs(os.path.join(WEIGHT_BACK_PATH, "train-weights"))
    for epoch in range(args.epoch_start, args.epoch_end, args.backup_freq):
        eval_model(epoch, args)
    # eval the last epoch
    eval_model(args.epoch_end - 1, args)


if __name__ == "__main__":
    main()
