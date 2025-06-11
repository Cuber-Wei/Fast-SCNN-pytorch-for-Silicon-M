import os
import logging
import torch
import torch.utils.data as data

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.metric import SegmentationMetric
from utils.visualize import get_color_pallete

from train import parse_args


class Evaluator(object):
    def __init__(self, args):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="./logs/evaluation.log",
            filemode="a+",
        )
        self.args = args
        # output folder
        self.outdir = "test_result"
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        # image transform
        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # dataset and dataloader
        val_dataset = get_segmentation_dataset(
            args.dataset, split="val", mode="testval", transform=input_transform
        )
        self.val_loader = data.DataLoader(
            dataset=val_dataset, batch_size=1, shuffle=False
        )
        # create network
        self.model = get_fast_scnn(
            args.dataset, aux=args.aux, pretrained=True, root=args.save_folder
        ).to(args.device)
        print("Finished loading model!")
        logging.info("Finished loading model!")
        logging.info(
            f"Model: {args.model}, Dataset: {args.dataset}, Aux: {args.aux}, Pretrained: {args.pretrained}, From Epoch: {args.idx}"
        )

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        self.model.eval()
        accu_sum = 0
        mIoU_sum = 0
        for i, (image, label) in enumerate(self.val_loader):
            image = image.to(self.args.device)

            outputs = self.model(image)

            pred = torch.argmax(outputs[0], 1)
            pred = pred.cpu().data.numpy()
            label = label.numpy()

            self.metric.update(pred, label)
            pixAcc, mIoU = self.metric.get()
            accu_sum += pixAcc
            mIoU_sum += mIoU
            print(
                "Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%"
                % (i + 1, pixAcc * 100, mIoU * 100)
            )
            logging.info(
                "Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%"
                % (i + 1, pixAcc * 100, mIoU * 100)
            )

            predict = pred.squeeze(0)
            mask = get_color_pallete(predict, self.args.dataset)
            mask.save(os.path.join(self.outdir, "seg_{}.png".format(i)))
        print(
            f"Average: pixAcc: {100 * accu_sum / len(self.val_loader):.3f}, mIoU: {100 * mIoU_sum / len(self.val_loader):.3f}"
        )
        logging.info(
            f"Epoch: {self.args.idx} Evaluation Finished! Average: validation pixAcc: {100 * accu_sum / len(self.val_loader):.3f}%, mIoU: {100 * mIoU_sum / len(self.val_loader):.3f}%"
        )
        with open("./evalRes.txt", "a") as f:
            f.write(
                f"Epoch: {self.args.idx}, Average: pixAcc: {100 * accu_sum / len(self.val_loader):.3f}%, mIoU: {100 * mIoU_sum / len(self.val_loader):.3f}%\n"
            )


if __name__ == "__main__":
    args = parse_args()
    evaluator = Evaluator(args)
    print("Testing model: ", args.model)
    evaluator.eval()
