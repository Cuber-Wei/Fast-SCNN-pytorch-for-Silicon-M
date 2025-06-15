import matplotlib.pyplot as plt
import os
from PIL import Image
from demo import demo

PATH = os.path.dirname(os.path.abspath(__file__))
input_pics = [
    "frankfurt_000001_060906_leftImg8bit.png",
    "frankfurt_000001_063045_leftImg8bit.png",
    "munster_000142_000019_leftImg8bit.png",
    "frankfurt_000000_010351_leftImg8bit.png",
    "lindau_000012_000019_leftImg8bit.png",
]

origin_paths = []
groundtruth_paths = []
output_paths = []

for image_name in input_pics:
    dirs = image_name.split("_")[0]
    origin_name = "_".join(image_name.split("_")[:3]) + "_leftImg8bit.png"
    file_name = "_".join(image_name.split("_")[:3]) + "_gtFine_color.png"
    origin_paths.append(
        os.path.join(PATH, "datasets/citys/leftImg8bit/val/", dirs, origin_name)
    )
    groundtruth_paths.append(
        os.path.join(PATH, "datasets/citys/gtFine/val/", dirs, file_name)
    )

# run demo.py for each image
for origin_path in origin_paths:
    demo(origin_path)
# add the result picture to the output_paths
for image_name in input_pics:
    output_paths.append(os.path.join(PATH, "test_result", image_name))

fig, axs = plt.subplots(5, 3, figsize=(15, 13))

for i in range(len(input_pics)):
    axs[i, 0].imshow(Image.open(origin_paths[i]).convert("RGB"))
    axs[i, 0].axis("off")

    axs[i, 1].imshow(Image.open(groundtruth_paths[i]).convert("RGB"))
    axs[i, 1].axis("off")

    axs[i, 2].imshow(Image.open(output_paths[i]).convert("RGB"))
    axs[i, 2].axis("off")

plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=0.2)
plt.subplots_adjust(
    left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.01, wspace=0.05
)
plt.savefig(os.path.join(PATH, "png/combined-result-visualization.png"), dpi=500)
