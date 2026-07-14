import argparse
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Save a selected CT slice from segmentation_output.npz"
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="segmentation_output.npz",
    )
    args = parser.parse_args()


    data = np.load('segmentation_output.npz')
    volume = data['volume_hu']
    masks = data['ventricle_masks']

    num_slices = len(volume)

    if args.slice is None:
        areas = masks.reshape(len(masks), -1).sum(axis=1)
        idx = int(np.argmax(areas))
        print(f"Auto-selected slice with largest ventricle area: {idx}")
    else:
        idx = args.slice

        if idx < 0 or idx >= num_slices:
            raise ValueError(
                f"슬라이드 인덱스 범위 초과"
            )


    img = volume[idx]
    img = np.clip(img, 0, 80)
    img = ((img - img.min()) / max(img.max() - img.min(), 1e-8) * 255).astype(np.uint8)

    cv2.imwrite(f'ct_slice_{idx}.png', img)
    print(f'Saved: ct_slice_{idx}.png')
    print(f"Volume shape: {volume.shape}")
    print(f"Selected slice: {idx}")

if __name__ == "__main__":
    main()