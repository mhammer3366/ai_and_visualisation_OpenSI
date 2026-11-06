"""
data.py - settings and lookup tables (shared setup)

Classes
    Config           All shared settings with their default values.
                     Change them in the first cell of the notebook.          (all questions)
    CountryMapper    Country codes <-> prompt names <-> ISO-3 codes, candidate lists,
                     and the paper's training-list flag.                     (all; Q2)
    OSV5MDataset     Downloads the OSV-5M test set, builds the pilot sample and
                     returns images + metadata by image id.                  (all)
    MetadataDecoder  Turns numeric metadata codes (climate, land cover,
                     driving side) into readable names.                      (Q4)
"""

import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
CONTINENT_NAMES = {
    "AF": "Africa", "AS": "Asia", "EU": "Europe", "NA": "North America",
    "SA": "South America", "OC": "Oceania", "AN": "Antarctica",
}

@dataclass
class Config:
    """All shared settings. Defaults are the agreed values for Phase I."""

    # --- general ---
    seed: int = 42
    device: str = "auto"                 # "auto" = GPU if available, otherwise CPU
    data_dir: str = "data"               # downloaded images, pilot ids, cached masks
    results_dir: str = "results"         # shared/, ai_pair/, human_pair/
    resources_dir: str = "resources"     # country_aliases.csv

    # --- sampling ---
    images_per_country: int = 50         # pilot sample size per country

    # --- models and inference ---
    streetclip_model: str = "geolocal/StreetCLIP"
    use_clip_baseline: bool = False      # True -> original CLIP is downloaded once (~1.7 GB)
    clip_model: str = "openai/clip-vit-large-patch14-336"
    prompt_template: str = "A Street View photo in {country}, {subregion}."
    candidate_list: str = "v1"           # "v1" or "all_test_locations"
    crop_mode: str = "center_crop"       # "center_crop", "pad" or "squash"
    batch_size: int = 32
    num_workers: int = 4                 # image-loading processes; set 0 if loading hangs
    half_precision: bool = True          # fp16 on GPU
    top_k: int = 10

    # --- segmentation and masking (Q3) ---
    segmentation_model: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    # Which Mapillary Vistas classes make up each element we can hide.
    # Names are matched ignoring case, spaces and punctuation.
    element_classes: dict = field(default_factory=lambda: {
        "road": ["Road", "Lane Marking - General", "Lane Marking - Crosswalk",
                 "Crosswalk - Plain", "Service Lane", "Bike Lane", "Parking",
                 "Manhole", "Pothole"],
        "vegetation": ["Vegetation", "Terrain"],
        "sky": ["Sky"],
        "buildings": ["Building", "Wall", "Bridge", "Tunnel"],
        "vehicles": ["Car", "Bus", "Truck", "Motorcycle", "Bicycle", "Caravan",
                     "Trailer", "Other Vehicle", "Wheeled Slow", "On Rails", "Boat"],
        "signs_poles": ["Traffic Sign (Front)", "Traffic Sign (Back)", "Traffic Sign Frame",
                        "Pole", "Utility Pole", "Street Light", "Traffic Light",
                        "Billboard", "Banner"],
    })
    ocr_languages: list = field(default_factory=lambda: ["en"])  # only boxes are used
    fill_mode: str = "gray"              # "gray", "blur" or "inpaint"
    blur_radius: int = 25
    mask_dilation_px: int = 5
    # Minimum share of the model's visible input an element must cover to be hidden.
    # Small elements (text, signs) get a much lower threshold.
    min_mask_area: dict = field(default_factory=lambda: {
        "default": 0.02, "text": 0.001, "signs_poles": 0.001,
    })
    num_random_controls: int = 3         # random same-size masks per real mask
    lama_url: str = ("https://github.com/enesmsahin/simple-lama-inpainting/"
                     "releases/download/v0.1.0/big-lama.pt")

    def check(self):
        """Catch typos in the settings that would otherwise fail much later."""
        choices = {
            "crop_mode": ["center_crop", "pad", "squash"],
            "candidate_list": ["v1", "all_test_locations"],
            "fill_mode": ["gray", "blur", "inpaint"],
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"{name} = {value!r}, must be one of {allowed}")
        for name in ["images_per_country", "batch_size", "top_k"]:
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self):
        return asdict(self)

    def show(self):
        """Print the scalar settings (the long class lists are left out)."""
        for name, value in self.to_dict().items():
            if not isinstance(value, (dict, list)):
                print(f"{name:22s} {value}")


# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------

class CountryMapper:
    """
    Reads resources/country_aliases.csv. Every country or territory has one
    canonical id: its ISO alpha-2 code (e.g. "DE"). That id is used in
    predictions, accuracy and maps; the prompt name is only used in the text
    given to the model.
    """

    def __init__(self, csv_path):
        # keep_default_na=False: otherwise pandas reads Namibia's code "NA" as missing
        self.table = pd.read_csv(csv_path, keep_default_na=False).set_index("iso2")

        # Lookup from any known spelling (ISO-2, ISO-3, prompt name) to ISO-2
        self._lookup = {}
        for iso2, row in self.table.iterrows():
            for key in (iso2, row["iso3"], row["prompt_name"]):
                self._lookup[key.lower()] = iso2

    def to_iso2(self, value):
        """Dataset label -> canonical id, or None if the label is unknown."""
        if not isinstance(value, str):
            return None
        return self._lookup.get(value.strip().lower())

    def prompt_name(self, iso2):
        return self.table.loc[iso2, "prompt_name"]

    def iso3(self, iso2):
        return self.table.loc[iso2, "iso3"]

    def sovereign_parent(self, iso2):
        """e.g. "PR" -> "US". Empty string for countries without a parent."""
        return self.table.loc[iso2, "sovereign_parent"]

    def in_training_list(self, iso2):
        """True if the StreetCLIP paper lists it as a training country (uncertain evidence)."""
        return bool(self.table.loc[iso2, "in_paper_training_list"])

    def candidates(self, version="v1", test_locations=None):
        """
        Fixed candidate list (ISO-2 ids, sorted).
          "v1"                 paper's 193 probing countries + training-list territories
          "all_test_locations" v1 + every location that occurs in the test set
        """
        v1 = set(self.table.index[self.table["in_candidates_v1"] == 1])
        if version == "v1":
            return sorted(v1)
        if version == "all_test_locations":
            return sorted(v1 | set(test_locations))
        raise ValueError(f"unknown candidate list {version!r}")

    def prompts(self, iso2_list, template):
        """Prompt text for each candidate, e.g. 'A Street View photo in Germany, Western Europe.'"""
        texts = []
        for c in iso2_list:
            continent = CONTINENT_NAMES[self.table.loc[c, "continent"]]
            subregion = self.table.loc[c, "world_subregion"] or continent   # empty only for Antarctic territories
            texts.append(template.format(country=self.prompt_name(c),
                                        continent=continent, subregion=subregion))
        return texts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OSV5MDataset:
    """
    The OSV-5M test split (~210k Mapillary images, 9.43 GB of zips + 116 MB csv).

    Typical use:
        dataset = OSV5MDataset(cfg, mapper)
        dataset.download()          # only downloads what is missing
        dataset.load()              # reads test.csv and indexes the images
        pilot_ids = dataset.build_pilot()
    """

    REPO_ID = "osv5m/osv5m"
    NUM_TEST_ZIPS = 5

    def __init__(self, cfg, mapper):
        self.cfg = cfg
        self.mapper = mapper
        self.root = Path(cfg.data_dir) / "osv5m_test"
        self.image_dir = self.root / "images" / "test"
        self.meta = None          # DataFrame, one row per image, index = image id
        self.image_paths = {}     # image id -> file path

    def download(self):
        """Download test.csv and the 5 image zips, extract them, delete the zips."""
        from huggingface_hub import hf_hub_download

        self.root.mkdir(parents=True, exist_ok=True)
        hf_hub_download(self.REPO_ID, "test.csv", repo_type="dataset", local_dir=self.root)

        for i in range(self.NUM_TEST_ZIPS):
            name = f"{i:02d}.zip"
            done_marker = self.image_dir / f"{i:02d}.done"
            if done_marker.exists():
                continue  # extracted in an earlier run

            print(f"Downloading and extracting {name} ...")
            zip_path = hf_hub_download(self.REPO_ID, name, subfolder="images/test",
                                       repo_type="dataset", local_dir=self.root)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(self.image_dir)
            Path(zip_path).unlink()   # free ~2 GB per zip
            done_marker.touch()

    def load(self):
        """Read test.csv, attach the canonical country id and index image files."""
        # Keep "NA" (Namibia) as text; only empty cells become missing values.
        meta = pd.read_csv(self.root / "test.csv", dtype={"id": str},
                           keep_default_na=False, na_values=[""])
        meta["country_id"] = meta["country"].map(self.mapper.to_iso2)

        self.image_paths = {p.stem: p for p in self.image_dir.rglob("*.jpg")}
        meta["has_image"] = meta["id"].isin(self.image_paths)

        self.meta = meta.set_index("id", drop=False)
        return self

    def unmapped_countries(self):
        """Dataset country labels that are missing from country_aliases.csv."""
        missing = self.meta[self.meta["country_id"].isna()]
        return missing["country"].value_counts()

    def usable(self):
        """Rows that have an image file and a known country."""
        return self.meta[self.meta["has_image"] & self.meta["country_id"].notna()]

    def all_ids(self):
        """Every usable image id (the full test set, for the main run)."""
        return self.usable()["id"].tolist()

    def test_locations(self):
        """All country ids that occur in the test set."""
        return sorted(self.usable()["country_id"].unique())
    
    def draw_pilot(self, n=None, seed=None):
        """
        Draw up to n images per country at random (fixed seed). Saves nothing.
        Countries with fewer images keep all of them (flagged as low_sample
        in country_counts).
        """
        n = n or self.cfg.images_per_country
        seed = self.cfg.seed if seed is None else seed

        shuffled = self.usable().sample(frac=1, random_state=seed)
        return shuffled.groupby("country_id").head(n)["id"].tolist()

    def build_pilot(self):
        """
        The shared pilot sample. Loaded from resources/pilot_ids.csv if that file
        exists, so everyone uses the same images. Otherwise it is drawn once and
        saved there (then commit the file). Delete the file to draw a new sample.
        """
        path = Path(self.cfg.resources_dir) / "pilot_ids.csv"
        if path.exists():
            return pd.read_csv(path, dtype={"id": str})["id"].tolist()

        pilot_ids = self.draw_pilot()
        pd.DataFrame({"id": pilot_ids}).to_csv(path, index=False)
        print(f"New pilot sample saved to {path} - commit this file.")
        return pilot_ids


    def country_counts(self, ids=None):
        """Images per country (for the given ids, or the whole test set)."""
        rows = self.usable() if ids is None else self.meta.loc[ids]
        counts = rows["country_id"].value_counts().rename("n_images").to_frame()
        counts["prompt_name"] = [self.mapper.prompt_name(c) for c in counts.index]
        counts["low_sample"] = counts["n_images"] < self.cfg.images_per_country
        counts["in_candidates_v1"] = [bool(self.mapper.table.loc[c, "in_candidates_v1"])
                                      for c in counts.index]
        counts["in_training_list"] = [self.mapper.in_training_list(c) for c in counts.index]
        return counts.rename_axis("country_id")

    def get_image(self, image_id):
        return Image.open(self.image_paths[image_id]).convert("RGB")

    def get_metadata(self, image_id):
        return self.meta.loc[image_id]


# ---------------------------------------------------------------------------
# Metadata codes
# ---------------------------------------------------------------------------

# Koppen-Geiger classes in the numbering of Beck et al. (2018), 0 = water / no data.
# OSV-5M reports 31 climate classes, which matches 0-30. Check against the data.
KOPPEN = {
    0: "water / no data",
    1: "Af tropical rainforest", 2: "Am tropical monsoon", 3: "Aw tropical savanna",
    4: "BWh hot desert", 5: "BWk cold desert", 6: "BSh hot steppe", 7: "BSk cold steppe",
    8: "Csa hot-summer Mediterranean", 9: "Csb warm-summer Mediterranean",
    10: "Csc cold-summer Mediterranean",
    11: "Cwa dry-winter humid subtropical", 12: "Cwb dry-winter subtropical highland",
    13: "Cwc dry-winter subpolar oceanic",
    14: "Cfa humid subtropical", 15: "Cfb oceanic", 16: "Cfc subpolar oceanic",
    17: "Dsa", 18: "Dsb", 19: "Dsc", 20: "Dsd",
    21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd",
    25: "Dfa hot-summer continental", 26: "Dfb warm-summer continental",
    27: "Dfc subarctic", 28: "Dfd extremely cold subarctic",
    29: "ET tundra", 30: "EF ice cap",
}

# FAO Global Land Cover SHARE classes (codes 1-11). OSV-5M names this source
# and reports 11 classes, but not its numbering. Check against the data.
GLC_SHARE = { 0: "no data",
    1: "artificial surfaces", 2: "cropland", 3: "grassland", 4: "tree covered",
    5: "shrubs", 6: "herbaceous, flooded", 7: "mangroves", 8: "sparse vegetation",
    9: "bare soil", 10: "snow and glaciers", 11: "water bodies",
}

# drive_side is binary (OSV-5M: left- or right-hand traffic). Checked on the data:
# all left-driving countries have 1, so 0 = right, 1 = left.
DRIVE_SIDE = {0: "right", 1: "left"}

# Left-driving places that OSV-5M wrongly marks as 0 (right) in every image.
# Found by counting drive_side per country; corrected to "left" in decode().
DRIVE_SIDE_FIXES = ["GB", "TZ", "BN", "FK", "VG", "VI", "CX", "NA"]


class MetadataDecoder:
    """
    Adds readable columns for the numeric metadata codes.
    The code tables above are taken from the sources OSV-5M names; their exact
    numbering in OSV-5M is not documented, so the notebook compares them with
    the values that actually occur.
    """

    @staticmethod
    def _name(code, table):
        if pd.isna(code):
            return None
        return table.get(int(round(code)), f"unknown code {code}")

    def climate_name(self, code):
        return self._name(code, KOPPEN)

    def land_cover_name(self, code):
        return self._name(code, GLC_SHARE)

    def drive_side_name(self, code):
        return self._name(code, DRIVE_SIDE)

    def decode(self, df):
        """Copy of df with climate_name, land_cover_name and drive_side_name added."""
        out = df.copy()
        out["climate_name"] = out["climate"].map(self.climate_name)
        out["land_cover_name"] = out["land_cover"].map(self.land_cover_name)
        out["drive_side_name"] = out["drive_side"].map(self.drive_side_name)
        out.loc[out["country_id"].isin(DRIVE_SIDE_FIXES), "drive_side_name"] = "left"
        return out
