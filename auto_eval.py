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


def main():
    args = parse_args()
    # get all the files in the weights-backup folder
    files = os.listdir(WEIGHT_BACK_PATH)
    if not os.path.exists(os.path.join(WEIGHT_BACK_PATH, "train-weights")):
        os.makedirs(os.path.join(WEIGHT_BACK_PATH, "train-weights"))
    for epoch in map(
        lambda s: s.split("-")[1].split(".")[0] if s.startswith("fast_scnn") else None,
        files,
    ):
        if epoch is not None:
            # backup: copy the weights file to the train-weights folder
            shutil.copyfile(
                os.path.join(WEIGHT_BACK_PATH, f"fast_scnn_citys-{epoch}.pth"),
                os.path.join(WEIGHT_BACK_PATH, "train-weights", f"fast_scnn_citys-{epoch}.pth"),
            )
            print("[INFO] Current Epoch", epoch)
            args.idx = epoch
            evaluator = Evaluator(args)
            # replace the weights file in the WEIGHTS_PATH
            os.replace(
                os.path.join(WEIGHT_BACK_PATH, f"fast_scnn_citys-{epoch}.pth"),
                os.path.join(WEIGHT_PATH, f"fast_scnn_citys.pth"),
            )
            # eval the model
            print("Testing model: ", args.model)
            evaluator.eval()
            del evaluator
            # rename the result dir
            os.rename("test_result", f"test_result_{epoch}")


if __name__ == "__main__":
    main()
