import os
import math
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import random
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.decomposition import PCA
from anomalib.metrics.aupro import _AUPRO as TM_AUPRO

from src.subspacead.config import get_args, parse_layer_indices, parse_grouped_layers
from src.subspacead.utils.common import (
    setup_logging,
    save_config,
    min_max_norm,
)
from src.subspacead.data.datasets import get_dataset_handler
from src.subspacead.core.extractor import FeatureExtractor
from src.subspacead.core.pca import PCAModel, KernelPCAModel
from src.subspacead.post_process.scoring import calculate_anomaly_scores, post_process_map
from src.subspacead.utils.viz import save_visualization, save_overlay_for_intro
from src.subspacead.post_process.specular import (
    specular_mask_torch,
    filter_specular_anomalies,
)
from src.subspacead.core.patching import process_image_patched, get_patch_coords
from src.subspacead.data.transforms import get_augmentation_transform

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AnomalyDetector:
    def __init__(self, args):
        self.args = args
        self.device = DEVICE
        self.extractor = None
        self.pca_model = None
        self.pca_params = None
        self.aug_transform = None
        self.categories = []
        self.results = []

        # Initialize logging and output directory
        self._setup_experiment()

        # Initialize components
        self._initialize_components()

        logging.info(f"Using device: {self.device}")

    def _setup_experiment(self):
        """Set up experiment directory and logging."""
        # Create run name
        run_name = self._create_run_name()

        # Set up output directory
        self.args.outdir = os.path.join(self.args.outdir, run_name)
        os.makedirs(self.args.outdir, exist_ok=True)

        # Set up logging
        setup_logging(self.args.outdir, not self.args.no_log_file)
        save_config(self.args)

        # Set up augmentations if needed
        self._setup_augmentations()

        # Get dataset categories
        self._get_categories()

    def _create_run_name(self) -> str:
        """Create a descriptive run name based on configuration."""
        run_name = (
            f"{self.args.dataset_name}_{self.args.agg_method}_"
            f"layers{''.join(self.args.layers.split(','))}_res{self.args.image_res}_"
            f"docrop{int(self.args.docrop)}"
        )

        if self.args.patch_size:
            run_name += f"_patch{self.args.patch_size}"
        if self.args.use_kernel_pca:
            run_name += f"_kpca-{self.args.kernel_pca_kernel}"
        if self.args.use_specular_filter:
            run_name += "_spec-filt"
        if self.args.bg_mask_method:
            run_name += f"_mask-{self.args.bg_mask_method}_thr-{self.args.mask_threshold_method}"
            if self.args.mask_threshold_method == "percentile":
                run_name += f"{self.args.percentile_threshold}"
            if self.args.bg_mask_method == "dino_saliency":
                run_name += f"_L{self.args.dino_saliency_layer}"

        run_name += f"_score-{self.args.score_method}"
        run_name += f"_clahe{int(self.args.use_clahe)}"
        run_name += f"_dropk{self.args.drop_k}"
        run_name += f"_model-{self.args.model_ckpt.split('/')[-1]}"
        run_name += (
            f"pca_ev{self.args.pca_ev}" if self.args.pca_ev is not None
            else f"_pca_dim{self.args.pca_dim}"
        )
        run_name += f"_i-score{self.args.img_score_agg}"

        # Add k-shot and augmentation info
        if self.args.k_shot is not None:
            run_name += f"_k{self.args.k_shot}"
            if self.args.aug_count > 0 and self.args.aug_list:
                aug_str = "".join(sorted([a[0] for a in self.args.aug_list]))
                run_name += f"_aug{self.args.aug_count}x{aug_str}"

        return run_name

    def _setup_augmentations(self):
        """Set up data augmentations if specified."""
        if self.args.k_shot is not None and self.args.aug_count > 0 and self.args.aug_list:
            self.aug_transform = get_augmentation_transform(
                self.args.aug_list, self.args.image_res
            )
            if not self.aug_transform.transforms:
                logging.warning(
                    "Augmentation specified but no valid transforms were created. "
                    "Disabling augmentations."
                )
                self.aug_transform = None

    def _get_categories(self):
        """Get dataset categories."""
        if self.args.categories:
            self.categories = self.args.categories
        else:
            self.categories = sorted([
                f.name for f in Path(self.args.dataset_path).iterdir()
                if f.is_dir() and f.name != "split_csv"
            ])
        logging.info(f"Found {len(self.categories)} categories: {self.categories}")

    def _initialize_components(self):
        """Initialize model components."""
        # Parse layer args
        self.layers = parse_layer_indices(self.args.layers)
        self.grouped_layers = (
            parse_grouped_layers(self.args.grouped_layers)
            if self.args.agg_method == "group" else []
        )

        # Initialize feature extractor
        self.extractor = FeatureExtractor(self.args.model_ckpt)

        logging.info(f"Current CUDA memory usage: {torch.cuda.memory_allocated() / 1024 / 1024}MB")

    def run(self):
        """Run the anomaly detection pipeline."""
        for category in self.categories:
            self._process_category(category)

        # Save final results
        self._save_results()

    def _process_category(self, category: str):
        """Process a single category."""
        logging.info(f"--- Processing Category: {category} ---")

        # Disable augmentation for specific categories if needed
        if category in self.args.no_aug_categories:
            logging.warning(f"Disabling augmentation for {category} category")
            self.aug_transform = None

        # Get dataset handler and paths
        handler = get_dataset_handler(self.args.dataset_name, self.args.dataset_path, category)
        train_paths = handler.get_train_paths()
        val_paths = handler.get_validation_paths()
        test_paths = handler.get_test_paths()

        logging.info(f"Current CUDA memory usage: {torch.cuda.memory_allocated() / 1024 / 1024}MB")

        # Apply debug limit if specified
        if self.args.debug_limit is not None:
            logging.warning(
                f"--- DEBUG MODE: Limiting validation and test sets to "
                f"{self.args.debug_limit} images ---"
            )
            if val_paths:
                val_paths = val_paths[:self.args.debug_limit]
            if test_paths:
                test_paths = test_paths[:self.args.debug_limit]

        # Skip if no training images
        if not train_paths:
            logging.warning(f"No training images found for {category}. Skipping.")
            return

        # Handle batched zero-shot mode
        if self.args.batched_zero_shot:
            logging.info(
                f"--- Batched 0-Shot Mode: Fitting PCA on {len(test_paths)} test images ---"
            )
            train_paths = test_paths.copy()
            val_paths = None

        # Apply k-shot sampling if specified
        if self.args.k_shot is not None:
            train_paths = self._apply_k_shot_sampling(train_paths)

        # Fit PCA model
        self._fit_pca_model(train_paths)

        # Determine thresholds if validation set exists
        thr_img, thr_px = self._determine_thresholds(val_paths, handler) if val_paths else (None, None)

        # Perform warm-up inference if test set exists
        if test_paths:
            self._warm_up_inference(test_paths)

        # Evaluate on test set
        self._evaluate_test_set(test_paths, handler, thr_img, thr_px, category)

    def _apply_k_shot_sampling(self, train_paths: List[str]) -> List[str]:
        """Apply k-shot sampling to training paths."""
        if self.args.k_shot > len(train_paths):
            logging.warning(
                f"Requested k_shot={self.args.k_shot} but only {len(train_paths)} "
                f"training images available. Using all {len(train_paths)}."
            )
            return train_paths

        logging.info(f"--- K-SHOT: Randomly sampling {self.args.k_shot} training images ---")
        random.shuffle(train_paths)
        sampled_paths = train_paths[:self.args.k_shot]

        for i, path in enumerate(sampled_paths):
            logging.info(f"  K-Shot image {i + 1}/{self.args.k_shot}: {Path(path).name}")

        return sampled_paths

    def _fit_pca_model(self, train_paths: List[str]):
        """Fit PCA model on training data."""
        if self.args.patch_size:
            self._fit_pca_model_with_patch(train_paths)
        else:
            self._fit_pca_model_without_patch(train_paths)

    def _fit_pca_model_with_patch(self, train_paths: List[str]):
        """Fit PCA model using patch-based approach."""
        if self.args.bg_mask_method == "pca_normality":
            logging.error(
                "PCA Normality mask is not compatible with --patch_size. "
                "Use 'dino_saliency' or no mask."
            )
            raise ValueError("Cannot use pca_normality mask with patch_size.")

        # Get feature dimensions from first image
        temp_img = Image.open(train_paths[0]).convert("RGB")
        temp_patch = temp_img.crop((0, 0, self.args.patch_size, self.args.patch_size))
        temp_tokens, (h_p, w_p), _ = self.extractor.extract_tokens(
            [temp_patch],
            self.args.image_res,
            self.layers,
            self.args.agg_method,
            self.grouped_layers,
            self.args.docrop,
            use_clahe=self.args.use_clahe,
            dino_saliency_layer=self.args.dino_saliency_layer,
        )
        feature_dim = temp_tokens.shape[-1]
        tokens_per_patch = h_p * w_p

        # Calculate total number of patches and tokens
        num_aug_multiplier = (1 + self.args.aug_count) if self.aug_transform else 1
        total_patches = 0
        num_batches = 0

        for path in train_paths:
            img = Image.open(path).convert("RGB")
            patch_coords = get_patch_coords(
                img.height, img.width, self.args.patch_size, self.args.patch_overlap
            )
            total_patches += len(patch_coords) * num_aug_multiplier
            num_batches += math.ceil(len(patch_coords) / self.args.batch_size) * num_aug_multiplier

        total_tokens = total_patches * tokens_per_patch

        logging.info(
            f"Feature dim: {feature_dim}, Tokens per patch: {tokens_per_patch}, "
            f"Base train patches: {total_patches // num_aug_multiplier}, "
            f"Total train patches (w/ aug): {total_patches}, Total train tokens: {total_tokens}"
        )

        # Create feature generator
        feature_generator = self._create_patched_feature_generator(train_paths, feature_dim)

        # Fit PCA model
        self._fit_pca(feature_generator, feature_dim, total_tokens, num_batches)

    def _create_patched_feature_generator(self, train_paths: List[str], feature_dim: int):
        """Create a generator for patched features."""
        def feature_generator():
            for path in train_paths:
                pil_img = Image.open(path).convert("RGB")

                # Create a list of images to process: original + augmentations
                images_to_process = [pil_img]
                if self.aug_transform:
                    for _ in range(self.args.aug_count):
                        images_to_process.append(self.aug_transform(pil_img))

                # Process each image (original + augmented)
                for img in images_to_process:
                    patch_coords = get_patch_coords(
                        img.height, img.width, self.args.patch_size, self.args.patch_overlap
                    )

                    for i in range(0, len(patch_coords), self.args.batch_size):
                        coord_batch = patch_coords[i:i + self.args.batch_size]
                        patch_batch = [img.crop(c) for c in coord_batch]

                        tokens_batch, _, saliency_masks_batch = self.extractor.extract_tokens(
                            patch_batch,
                            self.args.image_res,
                            self.layers,
                            self.args.agg_method,
                            self.grouped_layers,
                            self.args.docrop,
                            use_clahe=self.args.use_clahe,
                            dino_saliency_layer=self.args.dino_saliency_layer,
                        )
                        tokens_flat = tokens_batch.reshape(-1, feature_dim)

                        # Apply background masking if specified
                        if self.args.bg_mask_method == "dino_saliency":
                            tokens_flat = self._apply_saliency_mask(
                                tokens_flat, saliency_masks_batch
                            )

                        yield tokens_flat

        return feature_generator

    def _fit_pca_model_without_patch(self, train_paths: List[str]):
        """Fit PCA model without using patches."""
        logging.info(f"Current CUDA memory usage: {torch.cuda.memory_allocated() / 1024 / 1024}MB")

        # Get feature dimensions from first image
        temp_img = Image.open(train_paths[0]).convert("RGB")
        temp_tokens, (h_p, w_p), _ = self.extractor.extract_tokens(
            [temp_img],
            self.args.image_res,
            self.layers,
            self.args.agg_method,
            self.grouped_layers,
            self.args.docrop,
            use_clahe=self.args.use_clahe,
            dino_saliency_layer=self.args.dino_saliency_layer,
        )
        feature_dim = temp_tokens.shape[-1]
        num_aug_multiplier = (1 + self.args.aug_count) if self.aug_transform else 1
        total_train_images = len(train_paths) * num_aug_multiplier
        total_tokens = total_train_images * h_p * w_p

        logging.info(
            f"Feature dim: {feature_dim}, Tokens per image: {h_p * w_p}, "
            f"Base train images: {len(train_paths)}, "
            f"Total train images (w/ aug): {total_train_images}, Total train tokens: {total_tokens}"
        )

        # Create feature generator
        feature_generator = self._create_full_image_feature_generator(train_paths, feature_dim)
        num_batches = math.ceil(total_train_images / self.args.batch_size)

        # Fit PCA model
        self._fit_pca(feature_generator, feature_dim, total_tokens, num_batches)

    def _create_full_image_feature_generator(self, train_paths: List[str], feature_dim: int):
        """Create a generator for full image features."""
        def feature_generator():
            all_imgs_to_process = []

            # Collect all images to process
            for path in train_paths:
                pil_img = Image.open(path).convert("RGB")
                all_imgs_to_process.append(pil_img)
                if self.aug_transform:
                    for _ in range(self.args.aug_count):
                        all_imgs_to_process.append(self.aug_transform(pil_img))

            # Process images in batches
            for i in range(0, len(all_imgs_to_process), self.args.batch_size):
                img_batch = all_imgs_to_process[i:i + self.args.batch_size]

                tokens_batch, _, saliency_masks_batch = self.extractor.extract_tokens(
                    img_batch,
                    self.args.image_res,
                    self.layers,
                    self.args.agg_method,
                    self.grouped_layers,
                    self.args.docrop,
                    use_clahe=self.args.use_clahe,
                    dino_saliency_layer=self.args.dino_saliency_layer,
                )
                tokens_flat = tokens_batch.reshape(-1, feature_dim)

                # Apply background masking if specified
                if self.args.bg_mask_method == "dino_saliency":
                    tokens_flat = self._apply_saliency_mask(
                        tokens_flat, saliency_masks_batch
                    )

                yield tokens_flat

        return feature_generator

    def _apply_saliency_mask(self, tokens_flat: np.ndarray, saliency_masks_batch: np.ndarray) -> np.ndarray:
        """Apply saliency-based background masking to tokens."""
        masks_flat = saliency_masks_batch.reshape(-1)

        try:
            if self.args.mask_threshold_method == "percentile":
                threshold = np.percentile(masks_flat, self.args.percentile_threshold * 100)
                foreground_tokens = tokens_flat[masks_flat >= threshold]
            else:  # otsu
                norm_mask = cv2.normalize(
                    masks_flat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                _, binary_mask = cv2.threshold(
                    norm_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                foreground_tokens = tokens_flat[binary_mask.flatten() > 0]

            if foreground_tokens.shape[0] > 0:
                return foreground_tokens
            else:
                logging.warning("No foreground tokens found. Returning all tokens.")
                return tokens_flat
        except Exception as e:
            logging.warning(f"Saliency masking failed: {e}. Returning all tokens.")
            return tokens_flat

    def _fit_pca(self, feature_generator, feature_dim: int, total_tokens: int, num_batches: int):
        """Fit PCA model using the provided feature generator."""
        if self.args.use_kernel_pca:
            if self.args.bg_mask_method == "pca_normality":
                logging.error(
                    "PCA Normality mask is not compatible with Kernel PCA. "
                    "Use 'dino_saliency' or no mask."
                )
                raise ValueError("Cannot use pca_normality mask with use_kernel_pca.")

            logging.info("Collecting all features for Kernel PCA...")
            all_train_tokens = np.concatenate(
                list(tqdm(feature_generator(), desc="Feature Collection", total=num_batches))
            )

            self.pca_model = KernelPCAModel(
                k=self.args.pca_dim,
                kernel=self.args.kernel_pca_kernel,
                gamma=self.args.kernel_pca_gamma,
            )
            self.pca_params = self.pca_model.fit(all_train_tokens)
        else:
            self.pca_model = PCAModel(
                k=self.args.pca_dim, ev=self.args.pca_ev, whiten=self.args.whiten
            )
            self.pca_params = self.pca_model.fit(
                feature_generator, feature_dim, total_tokens, num_batches
            )

    def _determine_thresholds(self, val_paths: List[str], handler) -> Tuple[Optional[float], Optional[float]]:
        """Determine optimal thresholds for image and pixel-level anomaly scores."""
        logging.info(f"Collecting validation stats on {len(val_paths)} images for PR-optimal F1 thresholds...")

        val_img_scores, val_img_labels = [], []
        val_px_scores_normalized, val_px_gts = [], []

        val_iter = tqdm(val_paths, desc="Validating")

        for i in range(0, len(val_paths), self.args.batch_size):
            path_batch = val_paths[i:i + self.args.batch_size]
            pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
            is_anomaly_batch = [
                "good" not in p and "Normal" not in p for p in path_batch
            ]

            # Process validation batch
            batch_results = self._process_batch(
                pil_imgs, is_anomaly_batch, path_batch, handler, for_validation=True
            )

            # Collect results
            for j, (img_score, anomaly_map_normalized, gt_mask) in enumerate(batch_results):
                val_img_scores.append(img_score)
                val_img_labels.append(1 if is_anomaly_batch[j] else 0)

                if gt_mask is not None:
                    val_px_gts.extend(gt_mask.flatten().astype(np.uint8))
                    val_px_scores_normalized.extend(anomaly_map_normalized.flatten().astype(np.float32))

            val_iter.update(len(path_batch))

        # Determine thresholds
        target_img_fpr = getattr(self.args, "target_img_fpr", 0.05)
        target_px_fpr = getattr(self.args, "target_px_fpr", 0.05)

        thr_img, how_img = self._pick_threshold_with_fallback(
            val_img_labels, val_img_scores, target_img_fpr
        )

        val_px_scores_mm = np.array(val_px_scores_normalized)
        thr_px, how_px = self._pick_threshold_with_fallback(
            val_px_gts, val_px_scores_mm, target_px_fpr
        )

        if how_img == "none":
            logging.warning(
                "Validation image threshold degenerate and no negatives: image F1 will be NaN."
            )
        if how_px == "none":
            logging.warning(
                "Validation pixel threshold degenerate and no negatives: pixel F1 will be NaN."
            )

        logging.info(
            f"Chosen thresholds — Image: {thr_img if thr_img is not None else float('nan'):.6g} "
            f"({how_img}), Pixel: {thr_px if thr_px is not None else float('nan'):.6g} ({how_px})"
        )

        return thr_img, thr_px

    def _process_batch(
        self,
        pil_imgs: List[Image.Image],
        is_anomaly_batch: List[bool],
        path_batch: List[str],
        handler,
        for_validation: bool = False
    ) -> List[Tuple[float, np.ndarray, Optional[np.ndarray]]]:
        """Process a batch of images and return anomaly scores and maps."""
        results = []

        if self.args.patch_size:
            # Process with patch-based approach
            anomaly_maps_batch, saliency_maps_batch = process_image_patched(
                pil_imgs,
                self.extractor,
                self.pca_params,
                self.args,
                self.device,
                None,  # h_p will be determined inside process_image_patched
                None,  # w_p will be determined inside process_image_patched
                None,  # feature_dim will be determined inside process_image_patched
            )

            for j, anomaly_map_pre_specular in enumerate(anomaly_maps_batch):
                anomaly_map_final = anomaly_map_pre_specular

                # Apply specular filtering if specified
                if self.args.use_specular_filter:
                    anomaly_map_final = self._apply_specular_filter(
                        pil_imgs[j], anomaly_map_final
                    )

                # Calculate image-level score
                img_score = self._calculate_image_score(anomaly_map_final)

                # Get ground truth mask
                gt_mask = handler.get_ground_truth_mask(path_batch[j], pil_imgs[j].size) if for_validation else None
                if gt_mask is not None:
                    gt_mask = self._process_gt_mask(gt_mask, anomaly_map_final.shape)

                # Normalize anomaly map
                anomaly_map_normalized = min_max_norm(anomaly_map_final)

                results.append((img_score, anomaly_map_normalized, gt_mask))
        else:
            # Process with full-image approach
            tokens, (h_p, w_p), saliency_masks_batch = self.extractor.extract_tokens(
                pil_imgs,
                self.args.image_res,
                self.layers,
                self.args.agg_method,
                self.grouped_layers,
                self.args.docrop,
                use_clahe=self.args.use_clahe,
                dino_saliency_layer=self.args.dino_saliency_layer,
            )

            b, _, _, c = tokens.shape
            tokens_reshaped = tokens.reshape(b * h_p * w_p, c)

            # Calculate anomaly scores
            scores = calculate_anomaly_scores(
                tokens_reshaped,
                self.pca_params,
                self.args.score_method,
                self.args.drop_k,
            )
            anomaly_maps = scores.reshape(b, h_p, w_p)

            # Apply background masking if specified
            if self.args.bg_mask_method == "dino_saliency":
                anomaly_maps = self._apply_saliency_masking(
                    anomaly_maps, saliency_masks_batch, h_p, w_p
                )
            elif self.args.bg_mask_method == "pca_normality":
                anomaly_maps = self._apply_pca_normality_masking(
                    anomaly_maps, tokens, h_p, w_p, c
                )

            # Process each image in the batch
            for j in range(anomaly_maps.shape[0]):
                anomaly_map_pre_specular = post_process_map(anomaly_maps[j], self.args.image_res)
                anomaly_map_final = anomaly_map_pre_specular

                # Apply specular filtering if specified
                if self.args.use_specular_filter:
                    anomaly_map_final = self._apply_specular_filter(
                        pil_imgs[j], anomaly_map_final
                    )

                # Calculate image-level score
                img_score = self._calculate_image_score(anomaly_map_final)

                # Get ground truth mask
                gt_mask = None
                if for_validation:
                    gt_path_str = handler.get_ground_truth_path(path_batch[j])
                    if gt_path_str and os.path.exists(gt_path_str):
                        gt_mask_pil = Image.open(gt_path_str).convert("L")
                        if self.args.docrop:
                            resize_res = int(self.args.image_res / 0.875)
                            gt_mask_pil = TF.resize(
                                gt_mask_pil,
                                (resize_res, resize_res),
                                interpolation=TF.InterpolationMode.NEAREST,
                            )
                            gt_mask_pil = TF.center_crop(
                                gt_mask_pil, (self.args.image_res, self.args.image_res)
                            )
                        gt_mask_pil = TF.resize(
                            gt_mask_pil,
                            anomaly_map_final.shape,
                            interpolation=TF.InterpolationMode.NEAREST,
                        )
                        gt_mask = (np.array(gt_mask_pil) > 0).astype(np.uint8)

                # Normalize anomaly map
                anomaly_map_normalized = min_max_norm(anomaly_map_final)

                results.append((img_score, anomaly_map_normalized, gt_mask))

        return results

    def _apply_specular_filter(self, pil_img: Image.Image, anomaly_map: np.ndarray) -> np.ndarray:
        """Apply specular filtering to anomaly map."""
        img_tensor = TF.to_tensor(pil_img).unsqueeze(0).to(self.device)
        _, _, conf = specular_mask_torch(img_tensor, tau=self.args.specular_tau)
        conf = torch.nn.functional.interpolate(
            conf,
            size=anomaly_map.shape,
            mode="bilinear",
            align_corners=False,
        )
        conf_map = conf.squeeze().cpu().numpy()
        return filter_specular_anomalies(anomaly_map, conf_map).cpu().numpy()

    def _calculate_image_score(self, anomaly_map: np.ndarray) -> float:
        """Calculate image-level anomaly score from anomaly map."""
        if self.args.img_score_agg == "max":
            return float(np.max(anomaly_map))
        elif self.args.img_score_agg == "p99":
            return float(np.percentile(anomaly_map, 99))
        elif self.args.img_score_agg == "mtop5":
            return float(np.mean(np.sort(anomaly_map.flatten())[-5:]))
        elif self.args.img_score_agg == "mtop1p":
            return self._topk_mean(anomaly_map, frac=0.01)
        else:
            return float(np.mean(anomaly_map))

    def _topk_mean(self, arr: np.ndarray, frac: float = 0.01) -> float:
        """Calculate mean of top k elements in array."""
        flat = arr.ravel()
        k = max(1, int(len(flat) * frac))
        idx = np.argpartition(flat, -k)[-k:]
        return float(np.mean(flat[idx]))

    def _process_gt_mask(self, gt_mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Process ground truth mask to match target shape."""
        gt_mask = np.array(
            Image.fromarray((gt_mask.astype(np.uint8) * 255)).resize(
                target_shape, resample=Image.NEAREST
            )
        )
        return (gt_mask > 127).astype(np.uint8)

    def _apply_saliency_masking(
        self,
        anomaly_maps: np.ndarray,
        saliency_masks_batch: np.ndarray,
        h_p: int,
        w_p: int
    ) -> np.ndarray:
        """Apply saliency-based background masking to anomaly maps."""
        background_mask = np.zeros_like(anomaly_maps, dtype=bool)

        for j in range(anomaly_maps.shape[0]):
            saliency_map = saliency_masks_batch[j]
            try:
                if self.args.mask_threshold_method == "percentile":
                    threshold = np.percentile(saliency_map, self.args.percentile_threshold * 100)
                    background_mask[j] = saliency_map < threshold
                else:  # otsu
                    norm_mask = cv2.normalize(
                        saliency_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                    )
                    _, binary_mask = cv2.threshold(
                        norm_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    background_mask[j] = binary_mask == 0
            except Exception as e:
                logging.warning(
                    f"Saliency mask failed for image {j}: {e}. Skipping mask."
                )

        anomaly_maps[background_mask] = 0.0
        return anomaly_maps

    def _apply_pca_normality_masking(
        self,
        anomaly_maps: np.ndarray,
        tokens: np.ndarray,
        h_p: int,
        w_p: int,
        c: int
    ) -> np.ndarray:
        """Apply PCA normality-based background masking to anomaly maps."""
        threshold = 10.0
        kernel_size = 3
        border = 0.2
        grid_size = (h_p, w_p)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        background_mask_batch = np.zeros_like(anomaly_maps, dtype=bool)

        for j in range(anomaly_maps.shape[0]):
            img_features = tokens[j].reshape(-1, c)

            try:
                pca = PCA(n_components=1, svd_solver="randomized")
                first_pc = pca.fit_transform(img_features.astype(np.float32))

                mask = first_pc > threshold
                mask_2d = mask.reshape(grid_size)

                h_start, h_end = int(grid_size[0] * border), int(grid_size[0] * (1 - border))
                w_start, w_end = int(grid_size[1] * border), int(grid_size[1] * (1 - border))
                m = mask_2d[h_start:h_end, w_start:w_end]

                if m.sum() <= m.size * 0.35:
                    mask = -first_pc > threshold
                    mask_2d = mask.reshape(grid_size)

                # Post-process foreground mask
                mask_processed = cv2.dilate(mask_2d.astype(np.uint8), kernel).astype(bool)
                mask_processed = cv2.morphologyEx(
                    mask_processed.astype(np.uint8), cv2.MORPH_CLOSE, kernel
                ).astype(bool)

                # Invert the foreground mask to get the background mask
                background_mask_batch[j] = ~mask_processed
            except Exception as e:
                logging.warning(
                    f"PCA mask failed for image {j}: {e}. Skipping mask."
                )

        anomaly_maps[background_mask_batch] = 0.0
        return anomaly_maps

    def _warm_up_inference(self, test_paths: List[str]):
        """Perform warm-up inference run to initialize GPU and avoid timing issues."""
        logging.info("Performing warm-up inference run...")
        try:
            # Use the first test image for the warm-up
            dummy_img = [Image.open(test_paths[0]).convert("RGB")]

            if self.args.patch_size:
                # Warm-up the patch pipeline
                _ = process_image_patched(
                    dummy_img,
                    self.extractor,
                    self.pca_params,
                    self.args,
                    self.device,
                    None,  # h_p will be determined inside process_image_patched
                    None,  # w_p will be determined inside process_image_patched
                    None,  # feature_dim will be determined inside process_image_patched
                )
            else:
                # Warm-up the full-image pipeline
                _tokens, (_h, _w), _saliency = self.extractor.extract_tokens(
                    dummy_img,
                    self.args.image_res,
                    self.layers,
                    self.args.agg_method,
                    self.grouped_layers,
                    self.args.docrop,
                    use_clahe=self.args.use_clahe,
                    dino_saliency_layer=self.args.dino_saliency_layer,
                )
                # A minimal version of the scoring
                _scores = calculate_anomaly_scores(
                    _tokens.reshape(-1, _tokens.shape[-1]),
                    self.pca_params,
                    self.args.score_method,
                    self.args.drop_k,
                )
                if self.args.use_specular_filter and torch.cuda.is_available():
                    _ = filter_specular_anomalies(
                        torch.from_numpy(_scores).to(self.device),
                        torch.zeros_like(torch.from_numpy(_scores)).to(self.device),
                    )

            torch.cuda.synchronize(self.device)
            logging.info("Warm-up complete.")
        except Exception as e:
            logging.warning(f"Warm-up run failed: {e}. First timed run may be slow.")

    def _evaluate_test_set(
        self,
        test_paths: List[str],
        handler,
        thr_img: Optional[float],
        thr_px: Optional[float],
        category: str
    ):
        """Evaluate the model on the test set."""
        logging.info(f"Evaluating on {len(test_paths)} test images...")

        img_true, img_pred_f1 = [], []
        img_pred_auroc = []
        px_true_all = []
        px_pred_all_auroc = []
        px_pred_all_normalized = []
        anomalous_gt_masks = []
        anomalous_anomaly_maps = []
        vis_saved_count = 0
        all_inference_times = []

        logging.info(f"Number of test images: {len(test_paths)}")
        test_iter = tqdm(test_paths, desc=f"Testing {category}")

        for i in range(0, len(test_paths), self.args.batch_size):
            path_batch = test_paths[i:i + self.args.batch_size]
            pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
            is_anomaly_batch = [
                "good" not in str(p) and "Normal" not in str(p) for p in path_batch
            ]

            # Start timing
            torch.cuda.synchronize(self.device)
            start_time = time.perf_counter()

            # Process batch
            batch_results = self._process_batch(
                pil_imgs, is_anomaly_batch, path_batch, handler, for_validation=False
            )

            # End timing
            torch.cuda.synchronize(self.device)
            end_time = time.perf_counter()
            all_inference_times.append(end_time - start_time)

            # Collect results
            for j, (img_score, anomaly_map_normalized, gt_mask) in enumerate(batch_results):
                is_anomaly = is_anomaly_batch[j]
                path = path_batch[j]
                pil_img = pil_imgs[j]

                # Image-level metrics
                img_true.append(1 if is_anomaly else 0)
                img_pred_auroc.append(float(img_score))
                if thr_img is not None:
                    img_pred_f1.append(1 if img_score >= thr_img else 0)

                # Pixel-level metrics
                H, W = anomaly_map_normalized.shape
                gt_path_str = handler.get_ground_truth_path(path)

                if not gt_path_str or not os.path.exists(gt_path_str):
                    gt_mask = np.zeros((H, W), dtype=np.uint8)
                else:
                    gt_mask_pil = Image.open(gt_path_str).convert("L")
                    if self.args.docrop:
                        resize_res = int(self.args.image_res / 0.875)
                        gt_mask_pil = TF.resize(
                            gt_mask_pil,
                            (resize_res, resize_res),
                            interpolation=TF.InterpolationMode.NEAREST,
                        )
                        gt_mask_pil = TF.center_crop(
                            gt_mask_pil, (self.args.image_res, self.args.image_res)
                        )
                    gt_mask_pil = TF.resize(
                        gt_mask_pil,
                        (H, W),
                        interpolation=TF.InterpolationMode.NEAREST,
                    )
                    gt_mask = (np.array(gt_mask_pil) > 0).astype(np.uint8)

                px_true_all.extend(gt_mask.flatten().astype(np.uint8))
                px_pred_all_auroc.extend(anomaly_map_normalized.flatten().astype(np.float32))
                px_pred_all_normalized.extend(anomaly_map_normalized.flatten().astype(np.float32))

                # Store anomalous images for visualization and AUPRO calculation
                if is_anomaly:
                    anomalous_gt_masks.append(gt_mask)
                    anomalous_anomaly_maps.append(anomaly_map_normalized)

                    # Save visualization
                    if self.args.save_intro_overlays:
                        vis_img = pil_img
                        save_overlay_for_intro(
                            path, vis_img, anomaly_map_normalized, self.args.outdir, category
                        )

                    if vis_saved_count < self.args.vis_count:
                        vis_img = pil_img
                        if self.args.docrop and not self.args.patch_size:
                            resize_res = int(self.args.image_res / 0.875)
                            vis_img = TF.resize(
                                vis_img,
                                (resize_res, resize_res),
                                interpolation=TF.InterpolationMode.BICUBIC,
                            )
                            vis_img = TF.center_crop(vis_img, (self.args.image_res, self.args.image_res))

                        # Get saliency map for visualization if available
                        saliency_map_for_viz = None
                        if self.args.bg_mask_method == "dino_saliency" or self.args.bg_mask_method == "pca_normality":
                            saliency_map_for_viz = self._get_saliency_map_for_viz(
                                pil_img, anomaly_map_normalized.shape
                            )

                        save_visualization(
                            path,
                            vis_img,
                            gt_mask,
                            anomaly_map_normalized,
                            self.args.outdir,
                            category,
                            vis_saved_count,
                            saliency_mask=saliency_map_for_viz,
                        )
                        vis_saved_count += 1

            test_iter.update(len(path_batch))

        # Calculate and log timing results
        if all_inference_times:
            self._log_timing_results(all_inference_times, len(test_paths), category)

        # Calculate metrics
        self._calculate_and_log_metrics(
            img_true, img_pred_auroc, img_pred_f1,
            px_true_all, px_pred_all_auroc, px_pred_all_normalized,
            anomalous_gt_masks, anomalous_anomaly_maps,
            thr_img, thr_px, category
        )

    def _get_saliency_map_for_viz(self, pil_img: Image.Image, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Get saliency map for visualization."""
        try:
            _, _, saliency_masks_batch = self.extractor.extract_tokens(
                [pil_img],
                self.args.image_res,
                self.layers,
                self.args.agg_method,
                self.grouped_layers,
                self.args.docrop,
                use_clahe=self.args.use_clahe,
                dino_saliency_layer=self.args.dino_saliency_layer,
            )

            raw_mask_map = saliency_masks_batch[0]

            if self.args.bg_mask_method == "pca_normality":
                binary_mask = raw_mask_map
            elif self.args.bg_mask_method == "dino_saliency":
                if self.args.mask_threshold_method == "percentile":
                    threshold_val = np.percentile(raw_mask_map, self.args.percentile_threshold * 100)
                    binary_mask = (raw_mask_map >= threshold_val).astype(np.float32)
                else:  # otsu
                    norm_mask = cv2.normalize(
                        raw_mask_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                    )
                    _, binary_mask_u8 = cv2.threshold(
                        norm_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    binary_mask = (binary_mask_u8 > 0).astype(np.float32)

            return post_process_map(binary_mask, target_shape, blur=False)
        except Exception as e:
            logging.warning(f"Saliency mask processing failed for visualization: {e}.")
            return None

    def _log_timing_results(self, all_inference_times: List[float], num_images: int, category: str):
        """Calculate and log timing results."""
        times_arr = np.array(all_inference_times)
        total_time = np.sum(times_arr)
        avg_time_per_image = total_time / num_images
        images_per_second = 1.0 / avg_time_per_image

        logging.info(f"--- Timing Results for {category} ---")
        logging.info(f"Total test images: {num_images}")
        logging.info(
            f"Batch size: {self.args.batch_size} (Processed {len(all_inference_times)} batches)"
        )
        logging.info(f"Total inference time: {total_time:.4f} s")
        logging.info(f"Avg. time per image: {avg_time_per_image:.6f} s")
        logging.info(f"Images per second (FPS): {images_per_second:.2f}")

        # Report batch stats
        if len(times_arr) > 1:
            times_arr_stats = times_arr[1:]
            logging.info(f"Avg. time per batch (excl. 1st): {np.mean(times_arr_stats):.6f} s")
            logging.info(f"Median time per batch (excl. 1st): {np.median(times_arr_stats):.6f} s")
        else:
            logging.info(f"Avg. time per batch: {np.mean(times_arr):.6f} s")

    def _calculate_and_log_metrics(
        self,
        img_true: List[int],
        img_pred_auroc: List[float],
        img_pred_f1: List[int],
        px_true_all: List[int],
        px_pred_all_auroc: List[float],
        px_pred_all_normalized: List[float],
        anomalous_gt_masks: List[np.ndarray],
        anomalous_anomaly_maps: List[np.ndarray],
        thr_img: Optional[float],
        thr_px: Optional[float],
        category: str
    ):
        """Calculate and log evaluation metrics."""
        # Image-level metrics
        img_auroc = (
            roc_auc_score(img_true, img_pred_auroc)
            if len(np.unique(img_true)) > 1
            else np.nan
        )

        img_aupr = (
            average_precision_score(img_true, img_pred_auroc)
            if len(np.unique(img_true)) > 1
            else np.nan
        )

        img_f1 = f1_score(img_true, img_pred_f1) if thr_img is not None else np.nan

        # Pixel-level metrics
        px_true_arr = np.array(px_true_all, dtype=np.uint8)
        px_pred_arr_auroc = np.array(px_pred_all_auroc)
        px_pred_arr_normalized = np.array(px_pred_all_normalized)
        has_pos = (px_true_arr == 1).any()
        has_neg = (px_true_arr == 0).any()

        px_auroc = (
            roc_auc_score(px_true_arr, px_pred_arr_auroc)
            if (has_pos and has_neg)
            else np.nan
        )

        px_f1 = (
            f1_score(
                px_true_arr.astype(int),
                (px_pred_arr_normalized >= thr_px).astype(int),
            )
            if (thr_px is not None and has_pos)
            else np.nan
        )

        # AU-PRO metric
        if len(anomalous_gt_masks) > 0:
            preds_np = np.stack(anomalous_anomaly_maps).astype(np.float32)  # [N,H,W]
            gts_np = np.stack(anomalous_gt_masks).astype(np.uint8)  # [N,H_W]
            preds_t = (
                torch.from_numpy(preds_np).unsqueeze(1).to(torch.float32).to(self.device)
            )  # [N,1,H,W]
            gts_t = (
                torch.from_numpy(gts_np).unsqueeze(1).to(torch.bool).to(self.device)
            )  # [N,1,H,W]

            fpr_cap = getattr(self.args, "pro_integration_limit", 0.3)
            tm_metric = TM_AUPRO(fpr_limit=fpr_cap).to(self.device)
            au_pro = tm_metric(preds_t, gts_t).item()
        else:
            logging.warning(
                f"No anomalous images found in test set for {category}. AUPRO is not computable."
            )
            au_pro = np.nan

        # Log results
        logging.info(
            f"{category} Results | I-AUROC: {img_auroc:.4f} | I-AUPR: {img_aupr:.4f} | "
            f"P-AUROC: {px_auroc:.4f} | AU-PRO: {au_pro:.4f} | "
            f"I-F1: {img_f1:.4f} | P-F1: {px_f1:.4f}"
        )

        # Store results
        self.results.append(
            [category, img_auroc, img_aupr, px_auroc, au_pro, img_f1, px_f1]
        )

    def _pick_threshold_with_fallback(
        self,
        y_true: List[int],
        y_score: List[float],
        target_fpr: float
    ) -> Tuple[Optional[float], str]:
        """
        Try PR-optimal F1; if degenerate (single-class), fall back to negative-quantile.
        Returns (thr, how), where how ∈ {"pr", "quantile", "none"}.
        """
        thr_pr, _ = self._best_f1_threshold_from_scores(y_true, y_score)
        if thr_pr is not None:
            return thr_pr, "pr"

        thr_q = self._quantile_threshold_from_negatives(y_true, y_score, target_fpr)
        if thr_q is not None:
            return thr_q, "quantile"

        return None, "none"

    def _best_f1_threshold_from_scores(
        self,
        y_true: List[int],
        y_score: List[float]
    ) -> Tuple[Optional[float], float]:
        """Return threshold maximizing F1 on validation scores."""
        y_true = np.asarray(y_true).astype(np.uint8)
        y_score = np.asarray(y_score, dtype=np.float64)

        if y_true.size == 0 or y_score.size == 0 or (y_true.max() == y_true.min()):
            return None, 0.0

        p, r, t = precision_recall_curve(y_true, y_score)
        if t.size == 0:
            return None, 0.0

        f1 = (2 * p[:-1] * r[:-1]) / np.clip(p[:-1] + r[:-1], 1e-12, None)
        i = int(np.nanargmax(f1))

        return float(t[i]), float(f1[i])

    def _quantile_threshold_from_negatives(
        self,
        y_true: List[int],
        y_score: List[float],
        target_fpr: float = 0.01
    ) -> Optional[float]:
        """
        Fallback: pick threshold so that ~target_fpr of NEGATIVES exceed it.
        y_true in {0,1}, negatives are 0. Returns None if no negatives.
        """
        y_true = np.asarray(y_true).astype(np.uint8)
        y_score = np.asarray(y_score, dtype=np.float64)
        neg = y_score[y_true == 0]

        if neg.size == 0:
            return None

        q = np.clip(1.0 - float(target_fpr), 0.0, 1.0)
        return float(np.quantile(neg, q, interpolation="linear"))

    def _save_results(self):
        """Save final results to CSV."""
        df = pd.DataFrame(
            self.results,
            columns=[
                "Category",
                "Image AUROC",
                "Image AUPR",
                "Pixel AUROC",
                "AU-PRO",
                "Image F1",
                "Pixel F1",
            ],
        )

        if not df.empty and len(df) > 1:
            mean_values = df.mean(numeric_only=True)
            mean_row = pd.DataFrame(
                [["Average"] + mean_values.tolist()], columns=df.columns
            )
            df = pd.concat([df, mean_row], ignore_index=True)

        logging.info("\n--- Benchmark Final Results ---")
        logging.info("\n" + df.to_string(index=False, float_format="%.4f", na_rep="N/A"))

        results_path = os.path.join(self.args.outdir, "benchmark_results.csv")
        df.to_csv(results_path, index=False, float_format="%.4f")
        logging.info(f"\nResults saved to {results_path}")


def main():
    """Main entry point for the anomaly detection pipeline."""
    # Parse arguments
    args = get_args()

    # Initialize and run the anomaly detector
    detector = AnomalyDetector(args)
    detector.run()


if __name__ == "__main__":
    main()
