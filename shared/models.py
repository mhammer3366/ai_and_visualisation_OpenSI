"""
models.py - everything that loads a neural network (shared setup)

Classes
    GeoCLIPModel      StreetCLIP (or the original CLIP) as a zero-shot country
                      classifier: one score per candidate country.            (all; CLIP: Q2)
    PredictionRunner  Runs a GeoCLIPModel over many images and saves logits,
                      image embeddings and a per-image prediction table.      (all)
    Segmenter         Mask2Former (Mapillary Vistas) -> one boolean mask per
                      element (road, sky, ...).                               (Q3)
    TextDetector      EasyOCR text detection -> boolean text mask.            (Q3)
    Inpainter         LaMa inpainting; only loaded when fill_mode = "inpaint". (Q3)

Helpers
    get_device, free_gpu_memory, fit_image, fit_mask
"""

import gc
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

MODEL_INPUT_SIZE = 336                  # StreetCLIP / CLIP ViT-L/14-336 input size
CLIP_MEAN_RGB = (123, 117, 104)         # CLIP's normalization mean in 0-255: becomes 0 after normalizing

# Metadata columns copied from test.csv into every prediction table
METADATA_COLUMNS = ["latitude", "longitude", "country", "region", "sub-region", "city",
                    "land_cover", "road_index", "drive_side", "climate", "soil",
                    "dist_sea", "captured_at", "sequence"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(preference="auto"):
    """With "auto": NVIDIA GPU (cuda), else Apple GPU (mps), else CPU."""
    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preference


def free_gpu_memory():
    """Call after `del model` to give the GPU memory back."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _fit(image, mode, size, resample, pad_value):
    """Resize a PIL image to size x size using one of the three crop modes."""
    w, h = image.size
    if mode == "squash":
        return image.resize((size, size), resample)

    if mode == "center_crop":
        # Same as CLIP's own preprocessing: short side -> size, then cut the middle
        scale = size / min(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        resized = image.resize((new_w, new_h), resample)
        left, top = (new_w - size) // 2, (new_h - size) // 2
        return resized.crop((left, top, left + size, top + size))

    if mode == "pad":
        # Long side -> size, the rest is filled, so nothing is cut off
        scale = size / max(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        resized = image.resize((new_w, new_h), resample)
        canvas = Image.new(image.mode, (size, size), pad_value)
        canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
        return canvas

    raise ValueError(f"unknown crop mode {mode!r}")


def fit_image(image, mode, size=MODEL_INPUT_SIZE):
    """What the model actually sees: the image fitted into size x size."""
    return _fit(image, mode, size, Image.BICUBIC, CLIP_MEAN_RGB)


def fit_mask(mask, mode, size=MODEL_INPUT_SIZE):
    """Apply exactly the same geometry to a boolean mask (HxW numpy array)."""
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    fitted = _fit(mask_image, mode, size, Image.NEAREST, 0)
    return np.array(fitted) > 127


# ---------------------------------------------------------------------------
# StreetCLIP / CLIP
# ---------------------------------------------------------------------------

class GeoCLIPModel:
    """
    Zero-shot country classifier.

        model = GeoCLIPModel("geolocal/StreetCLIP", cfg)
        model.set_candidates(labels=["DE", "FR"], prompts=["A Street View photo in Germany.", ...])
        result = model.predict([image1, image2])   # logits, probs, embeddings

    How a score is computed: the image and every prompt are turned into vectors
    (embeddings) of length 768; the score is the cosine similarity times the
    model's learned scale. Softmax over the candidates gives probabilities.
    """

    def __init__(self, model_name, cfg):
        from transformers import CLIPModel, CLIPProcessor

        self.name = model_name
        self.crop_mode = cfg.crop_mode
        self.device = get_device(cfg.device)
        # fp16 on NVIDIA (cuda) and Apple (mps) GPUs; CPU uses fp32
        use_half = cfg.half_precision and self.device in ("cuda", "mps")
        self.dtype = torch.float16 if use_half else torch.float32

        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device).eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor

        self.labels = []          # candidate ids, e.g. ISO-2 codes
        self.prompts = []         # text given to the model for each candidate
        self.text_embeds = None   # (num_candidates, 768), normalized

    @torch.no_grad()
    def set_candidates(self, labels, prompts=None):
        """Encode the candidate prompts once. Without prompts, the labels are used as text."""
        prompts = list(labels) if prompts is None else list(prompts)
        tokens = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)

        # Same steps as CLIPModel.get_text_features, written out so it works across library versions
        pooled = self.model.text_model(**tokens).pooler_output
        embeds = self.model.text_projection(pooled)

        self.text_embeds = F.normalize(embeds.float(), dim=-1)
        self.labels = list(labels)
        self.prompts = prompts

    def prepare(self, images):
        """PIL images -> pixel tensor, using our crop mode instead of the processor's resizing."""
        fitted = [fit_image(img, self.crop_mode) for img in images]
        pixels = self.image_processor(fitted, do_resize=False, do_center_crop=False,
                                      return_tensors="pt")["pixel_values"]
        return pixels

    @torch.no_grad()
    def embed_images(self, pixel_values):
        """Pixel tensor -> normalized image embeddings (float32, on the model's device)."""
        pixel_values = pixel_values.to(self.device, self.dtype)
        pooled = self.model.vision_model(pixel_values=pixel_values).pooler_output
        embeds = self.model.visual_projection(pooled)
        return F.normalize(embeds.float(), dim=-1)

    @torch.no_grad()
    def logits_from_embeddings(self, image_embeds):
        """Scores for the current candidates. Also works on cached embeddings."""
        image_embeds = image_embeds.to(self.device).float()
        scale = self.model.logit_scale.exp().float()
        return scale * image_embeds @ self.text_embeds.T

    def predict(self, images):
        """List of PIL images -> dict with numpy arrays logits, probs and embeddings."""
        embeds = self.embed_images(self.prepare(images))
        logits = self.logits_from_embeddings(embeds)
        return {
            "logits": logits.cpu().numpy(),
            "probs": logits.softmax(dim=-1).cpu().numpy(),
            "embeddings": embeds.cpu().numpy(),
        }

    def top_k(self, probs, k=5):
        """For each image: [(label, probability), ...] sorted from most to least likely."""
        order = np.argsort(-probs, axis=1)[:, :k]
        return [[(self.labels[j], float(row[j])) for j in idx] for row, idx in zip(probs, order)]


class _FittedImages(torch.utils.data.Dataset):
    """Loads and preprocesses images in background workers (used by PredictionRunner)."""

    def __init__(self, paths, crop_mode, image_processor):
        self.paths = paths
        self.crop_mode = crop_mode
        self.image_processor = image_processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        fitted = fit_image(image, self.crop_mode)
        pixels = self.image_processor(fitted, do_resize=False, do_center_crop=False,
                                      return_tensors="pt")["pixel_values"]
        return pixels[0]


# ---------------------------------------------------------------------------
# Running predictions and saving them
# ---------------------------------------------------------------------------

class PredictionRunner:
    """
    Runs a model over a list of image ids and saves one run folder:

        results/shared/<run_name>/
            predictions.parquet   one row per image: true country, top-k, true-country
                                  probability and rank, correct, metadata, run settings
            logits.npy            all candidate scores, (num_images, num_candidates), float32
            embeddings.npy        image embeddings, (num_images, 768), float16
            ids.json              image ids in row order
            run_info.json         model, crop mode, prompt, candidate list, labels, ...

    Shared runs are read-only: an existing run folder is never overwritten
    unless overwrite=True is passed on purpose.
    """

    def __init__(self, cfg, dataset, mapper):
        self.cfg = cfg
        self.dataset = dataset
        self.mapper = mapper

    def run(self, model, image_ids, run_name, variant="original", out_dir=None, overwrite=False):
        out_dir = Path(out_dir) if out_dir else Path(self.cfg.results_dir) / "shared" / run_name
        if out_dir.exists() and not overwrite:
            raise FileExistsError(f"{out_dir} already exists - shared results are read-only")
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = [self.dataset.image_paths[i] for i in image_ids]
        loader = torch.utils.data.DataLoader(
            _FittedImages(paths, model.crop_mode, model.image_processor),
            batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)

        all_logits, all_embeds = [], []
        for pixels in tqdm(loader, desc=run_name):
            embeds = model.embed_images(pixels)
            all_logits.append(model.logits_from_embeddings(embeds).cpu().numpy())
            all_embeds.append(embeds.cpu().numpy().astype(np.float16))
        logits = np.concatenate(all_logits)
        embeds = np.concatenate(all_embeds)

        info = self._run_info(model, run_name, variant, len(image_ids))
        return self._save(out_dir, image_ids, logits, embeds, model.labels, info)

    def rescore(self, source_run, model, run_name, overwrite=False):
        """
        New candidate list or prompt without re-running the images:
        reuse the cached embeddings of source_run with the model's current candidates.
        The model must be the same model (and crop mode) that made the embeddings.
        """
        source = Path(self.cfg.results_dir) / "shared" / source_run
        out_dir = Path(self.cfg.results_dir) / "shared" / run_name
        if out_dir.exists() and not overwrite:
            raise FileExistsError(f"{out_dir} already exists - shared results are read-only")
        out_dir.mkdir(parents=True, exist_ok=True)

        image_ids = json.loads((source / "ids.json").read_text())
        embeds = np.load(source / "embeddings.npy")
        logits = model.logits_from_embeddings(torch.from_numpy(embeds)).cpu().numpy()

        info = self._run_info(model, run_name, "rescored", len(image_ids))
        info["rescored_from"] = source_run
        return self._save(out_dir, image_ids, logits, embeds, model.labels, info)

    def _run_info(self, model, run_name, variant, num_images):
        return {
            "run_name": run_name,
            "model": model.name,
            "crop_mode": model.crop_mode,
            "prompt_template": self.cfg.prompt_template,
            "candidate_list": self.cfg.candidate_list,
            "variant": variant,
            "num_images": num_images,
            "labels": model.labels,
        }

    def _save(self, out_dir, image_ids, logits, embeds, labels, info):
        np.save(out_dir / "logits.npy", logits.astype(np.float32))
        np.save(out_dir / "embeddings.npy", embeds.astype(np.float16))
        (out_dir / "ids.json").write_text(json.dumps(list(image_ids)))
        (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))

        table = self.build_table(image_ids, logits, labels, info)
        table.to_parquet(out_dir / "predictions.parquet", index=False)
        return table

    def build_table(self, image_ids, logits, labels, info):
        """One row per image with the prediction summary and the image's metadata."""
        k = min(self.cfg.top_k, len(labels))
        logits = torch.from_numpy(logits).float()
        probs = logits.softmax(dim=-1).numpy()
        logits = logits.numpy()
        label_index = {label: j for j, label in enumerate(labels)}

        meta = self.dataset.meta.loc[image_ids]
        table = pd.DataFrame({"id": list(image_ids), "country_id": meta["country_id"].values})

        # Top-k predictions
        order = np.argsort(-probs, axis=1)[:, :k]
        for rank in range(k):
            table[f"pred_{rank + 1}"] = [labels[j] for j in order[:, rank]]
            table[f"prob_{rank + 1}"] = probs[np.arange(len(probs)), order[:, rank]]

        # Where does the true country stand? Only possible if it is a candidate.
        true_index = table["country_id"].map(label_index)
        table["representable"] = true_index.notna()
        true_prob, true_rank = [], []
        for row, j in enumerate(true_index):
            if pd.isna(j):
                true_prob.append(np.nan)
                true_rank.append(np.nan)
            else:
                j = int(j)
                true_prob.append(probs[row, j])
                true_rank.append(int((logits[row] > logits[row, j]).sum()) + 1)
        table["true_prob"] = true_prob
        table["true_rank"] = true_rank

        # Images whose country is not a candidate are excluded (NA), not counted as wrong
        table["correct"] = pd.array(
            [pd.NA if not ok else bool(p == t)
             for ok, p, t in zip(table["representable"], table["pred_1"], table["country_id"])],
            dtype="boolean")

        for col in METADATA_COLUMNS:
            table[col] = meta[col].values
        for key in ["model", "crop_mode", "candidate_list", "prompt_template", "variant"]:
            table[key] = info[key]
        return table

    @staticmethod
    def load(run_dir):
        """Read a saved run: (table, logits, embeddings, info)."""
        run_dir = Path(run_dir)
        table = pd.read_parquet(run_dir / "predictions.parquet")
        logits = np.load(run_dir / "logits.npy")
        embeds = np.load(run_dir / "embeddings.npy")
        info = json.loads((run_dir / "run_info.json").read_text())
        return table, logits, embeds, info


# ---------------------------------------------------------------------------
# Segmentation (Q3)
# ---------------------------------------------------------------------------

def _simplify(name):
    """'Traffic Sign (Front)' -> 'trafficsignfront', so small spelling differences still match."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


class Segmenter:
    """
    Mask2Former trained on Mapillary Vistas (same image source as OSV-5M).
    Labels every pixel with one of ~65 street-scene classes and groups them into
    our elements (see Config.element_classes). Label maps are cached in
    data/masks/, so each image is segmented only once.
    """

    def __init__(self, cfg):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.device = get_device(cfg.device)
        self.processor = AutoImageProcessor.from_pretrained(cfg.segmentation_model)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            cfg.segmentation_model).to(self.device).eval()
        self.cache_dir = Path(cfg.data_dir) / "masks"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # element -> list of class ids
        id_by_name = {_simplify(name): int(i) for i, name in self.model.config.id2label.items()}
        self.element_ids = {}
        self.unmatched = []   # class names from the config that the model does not know
        for element, names in cfg.element_classes.items():
            ids = []
            for name in names:
                if _simplify(name) in id_by_name:
                    ids.append(id_by_name[_simplify(name)])
                else:
                    self.unmatched.append(name)
            self.element_ids[element] = ids

    @torch.no_grad()
    def label_map(self, image, image_id=None):
        """HxW array of class ids. Uses the cache when image_id is given."""
        cache = self.cache_dir / f"{image_id}_seg.npz" if image_id else None
        if cache and cache.exists():
            return np.load(cache)["labels"]

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        labels = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]
        labels = labels.cpu().numpy().astype(np.uint8)

        if cache:
            np.savez_compressed(cache, labels=labels)
        return labels

    def element_masks(self, image, image_id=None):
        """{element: HxW boolean mask} for every element in the config."""
        labels = self.label_map(image, image_id)
        return {element: np.isin(labels, ids) for element, ids in self.element_ids.items()}


class TextDetector:
    """
    EasyOCR text *detection* only: we need where text is, not what it says.
    Recognition confidence is not used, because with English recognition it is
    low on non-Latin scripts and would wrongly drop that text.
    """

    def __init__(self, cfg):
        import easyocr

        # gpu=True lets EasyOCR pick cuda or mps itself
        self.reader = easyocr.Reader(cfg.ocr_languages, gpu=get_device(cfg.device) != "cpu")
        self.cache_dir = Path(cfg.data_dir) / "masks"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def boxes(self, image):
        """Two lists: straight boxes [x_min, x_max, y_min, y_max] and tilted 4-point polygons."""
        straight, tilted = self.reader.detect(np.array(image))
        return straight[0], tilted[0]

    def text_mask(self, image, image_id=None):
        """HxW boolean mask covering all detected text. Cached when image_id is given."""
        cache = self.cache_dir / f"{image_id}_text.npz" if image_id else None
        if cache and cache.exists():
            return np.load(cache)["mask"]

        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        straight, tilted = self.boxes(image)
        for x_min, x_max, y_min, y_max in straight:
            x0, x1 = max(0, int(x_min)), min(w, int(x_max))
            y0, y1 = max(0, int(y_min)), min(h, int(y_max))
            mask[y0:y1, x0:x1] = 1
        for polygon in tilted:
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)

        mask = mask.astype(bool)
        if cache:
            np.savez_compressed(cache, mask=mask)
        return mask


class Inpainter:
    """
    LaMa ("big-lama") inpainting: fills a masked area with texture from its
    surroundings. Uses the same TorchScript file as the simple-lama-inpainting
    package; downloaded once (~200 MB) into the torch hub cache.
    """

    def __init__(self, cfg):
        self.device = get_device(cfg.device)
        cache = Path(torch.hub.get_dir()) / "checkpoints" / "big-lama.pt"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(cfg.lama_url, str(cache))
        self.model = torch.jit.load(str(cache), map_location=self.device).eval()

    @torch.no_grad()
    def inpaint(self, image, mask):
        """PIL image + HxW boolean mask -> PIL image with the masked area filled."""
        w, h = image.size
        # LaMa needs sides that are multiples of 8: pad by mirroring, crop afterwards
        pad_h, pad_w = (-h) % 8, (-w) % 8
        img = np.pad(np.array(image), ((0, pad_h), (0, pad_w), (0, 0)), mode="symmetric")
        msk = np.pad(mask.astype(np.float32), ((0, pad_h), (0, pad_w)), mode="symmetric")

        img_t = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(self.device)
        msk_t = torch.from_numpy(msk)[None, None].to(self.device)

        result = self.model(img_t, msk_t)[0].permute(1, 2, 0).cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype(np.uint8)[:h, :w]
        return Image.fromarray(result)
