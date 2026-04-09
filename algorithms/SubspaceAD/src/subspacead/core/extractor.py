import logging
import torch
from transformers import AutoImageProcessor, AutoModel
import cv2
import numpy as np
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor:
    """Encapsulates the feature extraction model and logic."""

    def __init__(self, model_ckpt: str, cache_dir: str = None):
        import time
        start_time = time.time()
        
        logging.info(f"[ModelLoad] {'='*50}")
        logging.info(f"[ModelLoad] Starting feature extraction model loading")
        logging.info(f"[ModelLoad] Model checkpoint: {model_ckpt}")
        logging.info(f"[ModelLoad] Target device: {DEVICE}")
        
        # 设置缓存目录
        if cache_dir:
            import os
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['HF_HOME'] = cache_dir
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            logging.info(f"[ModelLoad] Cache directory: {cache_dir}")
            logging.info(f"[ModelLoad] HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
            logging.info(f"[ModelLoad] TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")
        
        # 首先尝试从本地加载，失败则下载
        load_source = None
        try:
            logging.info(f"[ModelLoad] Phase 1: Attempting to load from local cache...")
            self.processor = AutoImageProcessor.from_pretrained(model_ckpt, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_ckpt, local_files_only=True).eval().to(DEVICE)
            load_source = "local_cache"
            logging.info(f"[ModelLoad] ✓ Model loaded from local cache successfully")
        except Exception as e:
            logging.warning(f"[ModelLoad] Local cache miss: {str(e)[:100]}")
            logging.info(f"[ModelLoad] Phase 2: Downloading from HuggingFace Hub...")
            self.processor = AutoImageProcessor.from_pretrained(model_ckpt, local_files_only=False)
            self.model = AutoModel.from_pretrained(model_ckpt, local_files_only=False).eval().to(DEVICE)
            load_source = "downloaded"
            logging.info(f"[ModelLoad] ✓ Model downloaded and loaded successfully")
        
        # 获取模型信息
        model_config = self.model.config
        num_layers = len(self.model.encoder.layer) if hasattr(self.model, 'encoder') else 'unknown'
        hidden_size = getattr(model_config, 'hidden_size', 'unknown')
        num_registers = getattr(model_config, 'num_register_tokens', 0)
        
        logging.info(f"[ModelLoad] Model architecture info:")
        logging.info(f"[ModelLoad]   - Model type: {getattr(model_config, 'model_type', 'unknown')}")
        logging.info(f"[ModelLoad]   - Hidden size: {hidden_size}")
        logging.info(f"[ModelLoad]   - Number of layers: {num_layers}")
        logging.info(f"[ModelLoad]   - Register tokens: {num_registers}")
        logging.info(f"[ModelLoad]   - Patch size: {getattr(model_config, 'patch_size', 'unknown')}")
        
        try:
            self.model.set_attn_implementation("eager")
            logging.info(f"[ModelLoad] Set attention implementation to 'eager'")
        except AttributeError:
            logging.warning(
                "[ModelLoad] Could not set attention implementation. Saliency masking might fail."
            )
        
        elapsed_time = time.time() - start_time
        logging.info(f"[ModelLoad] ✓ Model loading completed in {elapsed_time:.2f}s (source: {load_source})")
        logging.info(f"[ModelLoad] {'='*50}")

    def _apply_clahe(self, pil_imgs: list) -> list:
        """Applies CLAHE to a list of PIL images."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed_imgs = []
        for img in pil_imgs:
            img_np = np.array(img)
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_, a, b = cv2.split(img_lab)
            l_clahe = clahe.apply(l_)
            img_lab_clahe = cv2.merge((l_clahe, a, b))
            img_rgb_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)
            processed_imgs.append(Image.fromarray(img_rgb_clahe))
        return processed_imgs

    def _spatial_from_seq(
        self,
        seq_tokens: torch.Tensor,
        drop_front: int,
        n_expected: int,
        h_p: int,
        w_p: int,
    ) -> torch.Tensor:
        """Converts a sequence of tokens to a spatial (grid) format."""
        B, N, C = seq_tokens.shape
        tokens = seq_tokens[:, drop_front : drop_front + n_expected, :]
        return tokens.reshape(B, h_p, w_p, C)

    def _get_saliency_mask(
        self,
        attentions: tuple,
        dino_saliency_layer: int,
        num_reg: int,
        drop_front: int,
        n_expected: int,
        batch_size: int,
        h_p: int,
        w_p: int,
    ) -> np.ndarray:
        """Extracts the DINO saliency mask from attention weights."""
        if dino_saliency_layer < 0:
            dino_saliency_layer = len(attentions) + dino_saliency_layer

        if dino_saliency_layer >= len(attentions):
            logging.warning(
                f"DINO saliency layer {dino_saliency_layer} is out of bounds (0-{len(attentions)-1}). Defaulting to 0."
            )
            dino_saliency_layer = 0

        attn_map = attentions[dino_saliency_layer]
        if num_reg > 0:
            reg_attn_to_patches = attn_map[
                :, :, 1:drop_front, drop_front : drop_front + n_expected
            ]
            saliency_mask = reg_attn_to_patches.mean(dim=(1, 2))
        else:
            logging.info("No register tokens found. Using CLS token for saliency mask.")
            cls_attn_to_patches = attn_map[
                :, :, 0, drop_front : drop_front + n_expected
            ]
            saliency_mask = cls_attn_to_patches.mean(dim=1)

        return saliency_mask.reshape(batch_size, h_p, w_p).cpu().numpy()

    def _aggregate_layers(
        self,
        hidden_states: tuple,
        layers: list,
        grouped_layers: list,
        agg_method: str,
        drop_front: int,
        n_expected: int,
        h_p: int,
        w_p: int,
    ) -> np.ndarray:
        """Aggregates features from specified layers."""

        _spatial_converter = lambda x: self._spatial_from_seq(
            x, drop_front, n_expected, h_p, w_p
        )

        if agg_method == "group":
            if not grouped_layers:
                raise ValueError(
                    "Grouped layers must be provided for 'group' aggregation."
                )

            all_layer_indices = sorted(
                list(set(idx for group in grouped_layers for idx in group))
            )
            layer_tensors = {
                li: _spatial_converter(hidden_states[li]) for li in all_layer_indices
            }
            fused_groups = [
                torch.stack([layer_tensors[li] for li in group], dim=0).mean(dim=0)
                for group in grouped_layers
            ]
            fused = torch.cat(fused_groups, dim=-1)

        else:
            feats = [_spatial_converter(hidden_states[li]) for li in layers]
            if agg_method == "concat":
                fused = torch.cat(feats, dim=-1)
            elif agg_method == "mean":
                fused = torch.stack(feats, dim=0).mean(dim=0)
            else:
                raise ValueError(f"Unknown aggregation method: '{agg_method}'")

        return fused.cpu().numpy()

    @torch.no_grad()
    def extract_tokens(
        self,
        pil_imgs: list,
        res: int,
        layers: list,
        agg_method: str,
        grouped_layers: list = [],
        docrop: bool = False,
        use_clahe: bool = False,
        dino_saliency_layer: int = 0,
    ):
        """
        Extracts, aggregates features, and computes saliency from a batch of images.

        Returns:
            - fused_tokens (np.ndarray): The aggregated patch features.
            - grid_size (tuple): The (height, width) of the patch grid.
            - saliency_mask (np.ndarray): The DINO saliency mask.
        """

        # 1. Preprocessing
        if use_clahe:
            pil_imgs = self._apply_clahe(pil_imgs)

        if docrop:
            resize_res = int(res / 0.875)
            size = {"height": resize_res, "width": resize_res}
            crop_size = {"height": res, "width": res}
        else:
            size = {"height": res, "width": res}
            crop_size = {"height": res, "width": res}

        inputs = self.processor(
            images=pil_imgs,
            return_tensors="pt",
            do_resize=True,
            size=size,
            do_center_crop=docrop,
            crop_size=crop_size,
        ).to(DEVICE)

        # 2. Model Inference
        outputs = self.model(
            **inputs, output_hidden_states=True, output_attentions=True
        )
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        if attentions is None:
            raise ValueError(
                "Attention weights are None. Model may be using Flash Attention. "
                "Check transformers version or model compatibility."
            )

        # 3. Setup Parameters
        cfg = self.model.config
        ps = cfg.patch_size
        num_reg = getattr(cfg, "num_register_tokens", 0)
        drop_front = 1 + num_reg  # CLS token + register tokens
        h_p, w_p = res // ps, res // ps
        n_expected = h_p * w_p
        batch_size = inputs.pixel_values.shape[0]

        # 4. Saliency Mask Extraction
        saliency_mask = self._get_saliency_mask(
            attentions,
            dino_saliency_layer,
            num_reg,
            drop_front,
            n_expected,
            batch_size,
            h_p,
            w_p,
        )

        # 5. Feature Aggregation
        fused_tokens = self._aggregate_layers(
            hidden_states,
            layers,
            grouped_layers,
            agg_method,
            drop_front,
            n_expected,
            h_p,
            w_p,
        )

        return fused_tokens, (h_p, w_p), saliency_mask
