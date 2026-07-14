import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from medpy.filter.smoothing import anisotropic_diffusion
from scipy import ndimage
from tqdm import tqdm


class CTBrainSegmenter:
    def __init__(
        self, 
        skin_threshold: int = -100,
        skull_threshold: int = 150,
        brain_hu_range: tuple[int, int] = (20, 50),
        ventricle_hu_range: tuple[int, int] = (0, 20),
        kernel_size: tuple[int, int] = (5, 5),
        diffusion_iter: int = 10,
        diffusion_kappa: int = 30,
    ): 
        self.skin_threshold = skin_threshold
        self.skull_threshold = skull_threshold
        self.brain_hu_range = brain_hu_range
        self.ventricle_hu_range = ventricle_hu_range
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        self.diffusion_iter = diffusion_iter
        self.diffusion_kappa = diffusion_kappa
        
    
    def load_dicom_series(self, folder_path: str | Path) -> np.ndarray:
        dicom_files = sorted(Path(folder_path).glob('*.dcm'))
        if not dicom_files:
            raise FileNotFoundError(f"No .dcm files found in: {folder_path}")
        
        first_dcm = pydicom.dcmread(dicom_files[0])
        rescale_slope = getattr(first_dcm, 'RescaleSlope', 1)
        rescale_intercept = getattr(first_dcm, 'RescaleIntercept', 0)
        
        slices = []
        for dcm_file in tqdm(dicom_files, desc="Loading DICOM"):
            dcm = pydicom.dcmread(dcm_file)
            slope = float(getattr(dcm, "RescaleSlope", 1))
            intercept = float(getattr(dcm, "RescaleIntercept", 0))
            img_hu = dcm.pixel_array.astype(np.float32) * slope + intercept
            slices.append(img_hu)
            
        return np.stack(slices, axis=0)

    
    def segment_slice(self, img_hu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_filtered = anisotropic_diffusion(
            img_hu,
            niter=self.diffusion_iter,
            kappa=self.diffusion_kappa,
        )

        # 1. Head/Skin mask
        skin_mask = (img_filtered > self.skin_threshold).astype(np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.kernel, iterations=5)
        skin_mask = ndimage.binary_fill_holes(skin_mask).astype(np.uint8)
        skin_mask = self.largest_components_2d(skin_mask, num=1)
        
        skull_mask = img_filtered > 150
        brain_region = img_filtered.copy()
        brain_region[skull_mask] = 0 # 뼈 제거
        brain_mask = ((img_filtered > 25) & (img_filtered < 45)).astype(np.uint8)
        ventricle_mask = ((img_filtered > 0) & (img_filtered < 20)).astype(np.uint8)
        
        # 2. Skull mask
        skull_mask = (img_filtered > 150).astype(np.uint8)
        skull_mask = cv2.morphologyEx(skull_mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        
        # 3. Head mask excluding skull
        intracranial_region = skin_mask * (1 - skull_mask)
        intracranial_region = cv2.morphologyEx(
            intracranial_region, cv2.MORPH_CLOSE, self.kernel, iterations=5
        )
        intracranial_region = ndimage.binary_fill_holes(intracranial_region).astype(np.uint8)

        brain_low, brain_high = self.brain_hu_range
        brain_candidate = ((img_filtered > brain_low) & (img_filtered < brain_high)).astype(np.uint8)
        brain_mask = brain_candidate * intracranial_region # 뼈 영역 제외
        brain_mask = self.largest_components_2d(brain_mask, num=1)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, self.kernel, iterations=20)
        brain_mask = ndimage.binary_fill_holes(brain_mask).astype(np.uint8)
        # 외곽 smoothing
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, smooth_kernel,iterations=1,)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, smooth_kernel,iterations=1,)



        # 4. Ventricle mask
        vent_low, vent_high = self.ventricle_hu_range
        ventricle_candidate = ((img_filtered > vent_low) & (img_filtered < vent_high)).astype(np.uint8)
        ventricle_mask = ventricle_candidate * brain_mask
        ventricle_mask = self.largest_components_2d(ventricle_mask, num=2)
        ventricle_mask = cv2.morphologyEx(ventricle_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        ventricle_mask = ndimage.binary_fill_holes(ventricle_mask).astype(np.uint8)

        return skin_mask, brain_mask, ventricle_mask

    

    
    @staticmethod
    def largest_components_2d(mask: np.ndarray, num: int = 1) -> np.ndarray:
        """Keep the largest connected components in a 2D binary mask."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if num_labels <= 1:
            return mask.astype(np.uint8)

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_ids = (1 + np.argsort(areas)[-num:]).tolist()
        return np.isin(labels, largest_ids).astype(np.uint8)

    @staticmethod
    def largest_components_3d(mask_volume: np.ndarray, num: int = 1) -> np.ndarray:
        """Keep the largest connected components in a 3D binary volume."""
        labels, num_labels = ndimage.label(mask_volume.astype(bool))
        if num_labels <= 1:
            return mask_volume.astype(np.uint8)

        component_sizes = np.bincount(labels.ravel())
        component_sizes[0] = 0  # ignore background
        largest_ids = np.argsort(component_sizes)[-num:]
        return np.isin(labels, largest_ids).astype(np.uint8)



    def segment_volume(
        self,
        volume_hu: np.ndarray,
        apply_3d_cleanup: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        skin_masks, brain_masks, ventricle_masks = [], [], []

        for slice_hu in tqdm(volume_hu, desc="Segmenting"):
            skin_mask, brain_mask, ventricle_mask = self.segment_slice(slice_hu)
            skin_masks.append(skin_mask)
            brain_masks.append(brain_mask)
            ventricle_masks.append(ventricle_mask)

        skin_masks = np.stack(skin_masks).astype(np.uint8)
        brain_masks = np.stack(brain_masks).astype(np.uint8)
        ventricle_masks = np.stack(ventricle_masks).astype(np.uint8)

        if apply_3d_cleanup:
            skin_masks = self.largest_components_3d(skin_masks, num=1)
            brain_masks = self.largest_components_3d(brain_masks, num=1)
            ventricle_masks = self.largest_components_3d(ventricle_masks, num=2)

        return skin_masks, brain_masks, ventricle_masks



def create_overlay(img: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.3) -> np.ndarray:
    if img.ndim == 2:
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
    else:
        img_color = img.copy()

    overlay = img_color.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(img_color, 1 - alpha, overlay, alpha, 0)


def visualize_results(
    volume_hu: np.ndarray,
    skin_masks: np.ndarray,
    brain_masks: np.ndarray,
    ventricle_masks: np.ndarray,
    slice_indices: np.ndarray | None = None,
    save_path: str | Path = "segmentation_results.png",
) -> None:
    """Save and display representative segmentation results."""
    if slice_indices is None:
        slice_indices = np.linspace(0, len(volume_hu) - 1, 3, dtype=int)

    fig, axes = plt.subplots(len(slice_indices), 5, figsize=(15, 3 * len(slice_indices)))
    if len(slice_indices) == 1:
        axes = axes[np.newaxis, :]

    for row, slice_idx in enumerate(slice_indices):
        img_hu = volume_hu[slice_idx]
        skin_mask = skin_masks[slice_idx]
        brain_mask = brain_masks[slice_idx]
        ventricle_mask = ventricle_masks[slice_idx]

        panels = [
            (img_hu, "Original", "gray"),
            (skin_mask, "Skin/Head Mask", "gray"),
            (brain_mask, "Brain Mask", "gray"),
            (ventricle_mask, "Ventricle Mask", "gray"),
        ]

        for col, (image, title, cmap) in enumerate(panels):
            axes[row, col].imshow(image, cmap=cmap)
            axes[row, col].set_title(f"Slice {slice_idx}: {title}" if col == 0 else title)
            axes[row, col].axis("off")

        overlay = create_overlay(img_hu, skin_mask, color=(255, 100, 100), alpha=0.2)
        overlay = create_overlay(overlay, brain_mask, color=(0, 114, 255), alpha=0.3)
        overlay = create_overlay(overlay, ventricle_mask, color=(255, 255, 0), alpha=0.5)
        axes[row, 4].imshow(overlay)
        axes[row, 4].set_title("Overlay")
        axes[row, 4].axis("off") 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    
               
def main() -> None:
    parser = argparse.ArgumentParser(description="Classical OpenCV CT brain segmentation baseline")
    parser.add_argument("--folder", type=str, default="Brain CT (수술 전)", help="DICOM folder path")
    parser.add_argument("--slice", type=int, nargs="+", default=None, help="시각화할 슬라이스 번호")
    parser.add_argument("--output", type=str, default="segmentation_output.npz", help="Output .npz path")
    parser.add_argument("--figure", type=str, default=f"segmentation_results_.png", help="Output figure path")
    parser.add_argument("--no-3d-cleanup", action="store_true", help="Disable 3D connected-component cleanup")
    args = parser.parse_args()

    if len(args.slice) == 1:
        args.figure = f"segmentation_results_{args.slice[0]}.png"


    segmenter = CTBrainSegmenter(kernel_size=(5, 5))
    volume_hu = segmenter.load_dicom_series(args.folder)
    skin_masks, brain_masks, ventricle_masks = segmenter.segment_volume(
        volume_hu,
        apply_3d_cleanup=not args.no_3d_cleanup,
    )

    visualize_results(volume_hu, skin_masks, brain_masks, ventricle_masks, slice_indices=args.slice, save_path=args.figure)

    np.savez_compressed(
        args.output,
        volume_hu=volume_hu,
        skin_masks=skin_masks,
        brain_masks=brain_masks,
        ventricle_masks=ventricle_masks,
    )


if __name__ == "__main__":
    main()
