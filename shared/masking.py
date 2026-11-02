"""
masking.py - hiding image elements (shared setup, Q3)

Class
    MaskApplier   Prepares a mask (dilation, visible-area check against what the
                  model actually sees after cropping), fills the masked area
                  (gray, blur or LaMa inpainting) and creates random same-size
                  masks as controls. Skipped masks are logged.

The masking experiments themselves (which images, which elements, how to
compare results) are built separately by each pair.
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .models import CLIP_MEAN_RGB, fit_mask


class MaskApplier:
    """
        applier = MaskApplier(cfg)                      # gray / blur
        applier = MaskApplier(cfg, Inpainter(cfg))      # also inpaint

        result = applier.apply(image, masks["road"], element="road", image_id="123")
        if result["image"] is not None:
            ...  # result["image"] is the masked PIL image
    """

    def __init__(self, cfg, inpainter=None):
        self.cfg = cfg
        self.inpainter = inpainter
        self.skipped = []   # one dict per mask that was too small to use

    # --- preparing the mask -------------------------------------------------

    def dilate(self, mask):
        """Grow the mask by mask_dilation_px pixels to cover leftover edges."""
        r = self.cfg.mask_dilation_px
        if r <= 0:
            return mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0

    def visible_fraction(self, mask):
        """
        Share of the model's input (after crop_mode) covered by the mask.
        With center_crop, mask parts at the left/right edges are cut off and
        therefore do not count: they were never seen by the model.
        """
        return float(fit_mask(mask, self.cfg.crop_mode).mean())

    def threshold(self, element):
        return self.cfg.min_mask_area.get(element, self.cfg.min_mask_area["default"])

    # --- filling ------------------------------------------------------------

    def fill(self, image, mask, mode=None):
        """Return a copy of image with the masked pixels replaced."""
        mode = mode or self.cfg.fill_mode
        pixels = np.array(image)

        if mode == "gray":
            # CLIP's mean colour: becomes exactly 0 after the model's normalization
            pixels[mask] = CLIP_MEAN_RGB
            return Image.fromarray(pixels)

        if mode == "blur":
            blurred = cv2.GaussianBlur(pixels, (0, 0), sigmaX=self.cfg.blur_radius)
            pixels[mask] = blurred[mask]
            return Image.fromarray(pixels)

        if mode == "inpaint":
            if self.inpainter is None:
                raise ValueError("fill_mode 'inpaint' needs MaskApplier(cfg, Inpainter(cfg))")
            return self.inpainter.inpaint(image, mask)

        raise ValueError(f"unknown fill mode {mode!r}")

    # --- the main step ------------------------------------------------------

    def apply(self, image, mask, element, image_id=None, mode=None):
        """
        Dilate, check the visible area, fill.

        Returns a dict:
            image             masked PIL image, or None if the mask was skipped
            mask              the dilated mask that was used
            visible_fraction  share of the model's input that is hidden
        """
        mask = self.dilate(mask)
        visible = self.visible_fraction(mask)

        if visible < self.threshold(element):
            self.skipped.append({"image_id": image_id, "element": element,
                                 "visible_fraction": visible,
                                 "threshold": self.threshold(element)})
            return {"image": None, "mask": mask, "visible_fraction": visible}

        return {"image": self.fill(image, mask, mode), "mask": mask, "visible_fraction": visible}

    def random_controls(self, mask, n=None, seed=None):
        """
        n masks with exactly the same shape and area as `mask`, moved to random
        positions (wrapping around the image edges). Hiding these shows how much
        a change comes simply from hiding that many pixels.
        """
        n = n or self.cfg.num_random_controls
        rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        h, w = mask.shape
        return [np.roll(mask, (int(rng.integers(h)), int(rng.integers(w))), axis=(0, 1))
                for _ in range(n)]

    def apply_controls(self, image, used_mask, n=None, seed=None, mode=None):
        """
        Fill the random controls of a mask that was already used (i.e. already
        dilated by apply()). Returns a list of dicts like apply().
        The full-image area always matches; the visible fraction can differ when
        a control is moved into a part that the crop cuts off, so it is reported.
        """
        results = []
        for control in self.random_controls(used_mask, n, seed):
            results.append({"image": self.fill(image, control, mode), "mask": control,
                            "visible_fraction": self.visible_fraction(control)})
        return results

    def skipped_log(self):
        """All skipped masks as a table."""
        return pd.DataFrame(self.skipped, columns=["image_id", "element",
                                                   "visible_fraction", "threshold"])
