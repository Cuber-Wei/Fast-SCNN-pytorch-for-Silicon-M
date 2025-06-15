import os
import argparse
import time
import shutil
import logging

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
from utils.lr_scheduler import LRScheduler
from utils.metric import SegmentationMetric


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description="Fast-SCNN on PyTorch")
    # model and dataset
    parser.add_argument(
        "--model", type=str, default="fast_scnn", help="model name (default: fast_scnn)"
    )
    parser.add_argument(
        "--dataset", type=str, default="citys", help="dataset name (default: citys)"
    )
    parser.add_argument("--base-size", type=int, default=1024, help="base image size")
    parser.add_argument("--crop-size", type=int, default=768, help="crop image size")
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="dataset train split (default: train)",
    )
    # training hyper params
    parser.add_argument(
        "--aux", action="store_true", default=False, help="Auxiliary loss"
    )
    parser.add_argument(
        "--aux-weight", type=float, default=0.4, help="auxiliary loss weight"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=160,
        metavar="N",
        help="number of epochs to train (default: 160)",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        metavar="N",
        help="start epochs (default:0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        metavar="N",
        help="input batch size for training (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        metavar="LR",
        help="learning rate (default: 1e-2)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        metavar="M",
        help="w-decay (default: 1e-4)",
    )
    # checking point
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="put the path to resuming file if needed",
    )
    parser.add_argument(
        "--save-folder",
        default="./weights",
        help="Directory for saving checkpoint models",
    )
    parser.add_argument(
        "--backup-freq",
        type=int,
        default=5,
        help="save frequency (default: 5)",
    )
    # evaluation only
    parser.add_argument(
        "--eval", action="store_true", default=False, help="evaluation only"
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        default=True,
        help="skip validation during training",
    )
    # epoch index, for specific epoch evaluation
    parser.add_argument("--idx", type=int, default=0, help="index of the training")
    parser.add_argument("--epoch-start", type=int, default=0, help="start eval epoch")
    parser.add_argument("--epoch-end", type=int, default=160, help="end eval epoch")
    # the parser
    args = parser.parse_args()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        cudnn.benchmark = True
    args.device = device
    print(f"Using device: {device}")
    return args


class Trainer(object):
    def __init__(self, args):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="./logs/train.log",
            filemode="w",
        )
        self.args = args
        # image transform
        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # dataset and dataloader
        data_kwargs = {
            "transform": input_transform,
            "base_size": args.base_size,
            "crop_size": args.crop_size,
        }
        train_dataset = get_segmentation_dataset(
            args.dataset, split=args.train_split, mode="train", **data_kwargs
        )
        val_dataset = get_segmentation_dataset(
            args.dataset, split="val", mode="val", **data_kwargs
        )
        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.val_loader = data.DataLoader(
            dataset=val_dataset, batch_size=1, shuffle=False
        )

        # create network
        self.model = get_fast_scnn(dataset=args.dataset, aux=args.aux)
        if args.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2])
        self.model.to(args.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == ".pkl" or ".pth", (
                    "Sorry only .pth and .pkl files supported."
                )
                print("Resuming training, loading {}...".format(args.resume))
                self.model.load_state_dict(
                    torch.load(args.resume, map_location=args.device)
                )

        # create criterion
        self.criterion = MixSoftmaxCrossEntropyOHEMLoss(
            aux=args.aux, aux_weight=args.aux_weight, ignore_index=-1
        ).to(args.device)

        # optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        # lr scheduling
        self.lr_scheduler = LRScheduler(
            mode="poly",
            base_lr=args.lr,
            nepochs=args.epochs,
            iters_per_epoch=len(self.train_loader),
            power=0.9,
        )

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0

    def train(self):
        cur_iters = 0
        start_time = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()

            for i, (images, targets) in enumerate(self.train_loader):
                cur_lr = self.lr_scheduler(cur_iters)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = cur_lr

                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cur_iters += 1
                if cur_iters % 10 == 0:
                    print(
                        "Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f"
                        % (
                            epoch,
                            args.epochs,
                            i + 1,
                            len(self.train_loader),
                            time.time() - start_time,
                            cur_lr,
                            loss.item(),
                        )
                    )
                    logging.info(
                        "Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f"
                        % (
                            epoch,
                            args.epochs,
                            i + 1,
                            len(self.train_loader),
                            time.time() - start_time,
                            cur_lr,
                            loss.item(),
                        )
                    )

            if self.args.no_val:
                # save every epoch as checkpoint
                save_checkpoint(self.model, self.args, is_best=False)
                if epoch % self.args.backup_freq == 0 or epoch == self.args.epochs - 1:
                    # save every 5 epochs for evalation
                    torch.save(
                        self.model.state_dict(),
                        f"./weights-backup/{self.args.model}_{self.args.dataset}-{epoch}.pth",
                    )
                    print(
                        f"Saved model to ./weights-backup/{self.args.model}_{self.args.dataset}-{epoch}.pth"
                    )
            else:
                self.validation(epoch)

        save_checkpoint(self.model, self.args, is_best=False)

    def validation(self, epoch):
        is_best = False
        self.metric.reset()
        self.model.eval()
        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.args.device)

            outputs = self.model(image)
            pred = torch.argmax(outputs[0], 1)
            pred = pred.cpu().data.numpy()
            self.metric.update(pred, target.numpy())
            pixAcc, mIoU = self.metric.get()
            print(
                "Epoch %d, Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%"
                % (epoch, i + 1, pixAcc * 100, mIoU * 100)
            )

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, is_best)


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = "{}_{}.pth".format(args.model, args.dataset)
    save_path = os.path.join(directory, filename)
    torch.save(model.state_dict(), save_path)
    if is_best:
        best_filename = "{}_{}_best_model.pth".format(args.model, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print("Evaluation model: ", args.resume)
        trainer.validation(args.start_epoch)
    else:
        print("Starting Epoch: %d, Total Epochs: %d" % (args.start_epoch, args.epochs))
        trainer.train()
