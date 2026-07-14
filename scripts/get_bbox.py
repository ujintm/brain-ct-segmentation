import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--slice",
    type=int,
    required=True,
)
parser.add_argument(
    "--input",
    default="segmentation_output.npz",
)
parser.add_argument(
    "--pad",
    type=int,
    default=10,
)

args = parser.parse_args()

data = np.load(args.input)
masks = data["ventricle_masks"]

if args.slice < 0 or args.slice >= len(masks):
    raise ValueError(f"slice는 0~{len(masks)-1} 사이여야")

mask = masks[args.slice]
area = int(mask.sum())

if area == 0:
    raise RuntimeError(f"Slice {args.slice}에는 ventricle mask가 없음")

ys, xs = np.where(mask > 0)

h, w = mask.shape

box = [
    max(0, int(xs.min()) - args.pad),
    max(0, int(ys.min()) - args.pad),
    min(w - 1, int(xs.max()) + args.pad),
    min(h - 1, int(ys.max()) + args.pad),
]

print("slice:", args.slice)
print("area:", area)
print("box:", *box)
print(f'--box "[{box[0]},{box[1]},{box[2]},{box[3]}]"')

