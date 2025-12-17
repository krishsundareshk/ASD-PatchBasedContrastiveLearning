# astra_attn_patch_dataset.py

import os
from glob import glob
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

ALL_MACHINE_TYPES = ["ToyCar","ToyTrain","bearing","valve","fan","gearbox","slider"]
USE_ATTR = {"fan","ToyCar","valve","gearbox"}

class ASTRA_AttnPatchRGBDataset(Dataset):
    """
    Base dataset (train/supplement) returning:
      - 'patches_1': (N,3,patch,patch)
      - 'patches_2': (N,3,patch,patch)  (only for train)
      - 'attrs'    : (global_attr_dim,)
      - 'filename' : str
    """
    def __init__(self, root_dir, machine_type, split="train",
                 patch_size=32, stride=16, max_patches=None,
                 global_attr_dim=0):
        assert machine_type in ALL_MACHINE_TYPES
        self.root_dir = root_dir
        self.machine_type = machine_type
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches = max_patches
        self.global_attr_dim = global_attr_dim

        # collect all PNGs
        folder = os.path.join(root_dir, machine_type, f"{split}RGB")
        self.samples = sorted(glob(os.path.join(folder, "*.png")))

        # load and encode attributes if present
        self.machine_attr_dim = 0
        self.attr_map = {}
        if machine_type in USE_ATTR:
            csv_p = os.path.join(root_dir, machine_type, "attributes_00.csv")
            if os.path.isfile(csv_p):
                df = pd.read_csv(csv_p)
                fcol = "filename" if "filename" in df.columns else "file_name"
                df["basename"] = df[fcol].astype(str).apply(os.path.basename)
                cols = [c for c in df.columns if c not in {fcol,"basename"}]
                for c in cols:
                    if not pd.api.types.is_numeric_dtype(df[c]):
                        df[c] = df[c].astype("category").cat.codes
                self.machine_attr_dim = len(cols)
                for _, row in df.iterrows():
                    vec = torch.tensor(row[cols].to_numpy(dtype=float), dtype=torch.float32)
                    self.attr_map[row["basename"]] = vec

        # transforms
        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                # transforms.Normalize([0.5]*3,[0.5]*3)
                # Dinov2 normalization values
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3,[0.5]*3),
            ])

    def extract_patches(self, img: torch.Tensor):
        unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.stride)
        patches = unfold(img.unsqueeze(0)).squeeze(0).T
        return patches.view(-1, 3, self.patch_size, self.patch_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        fname = os.path.basename(path)
        img = Image.open(path).convert("RGB")

        v1 = self.transform(img)
        p1 = self.extract_patches(v1)
        if self.max_patches and p1.size(0) > self.max_patches:
            sel = torch.randperm(p1.size(0))[:self.max_patches]
            p1 = p1[sel]

        # build attrs vector (possibly zero)
        attrs = self.attr_map.get(fname, torch.zeros(self.machine_attr_dim))
        if self.global_attr_dim > self.machine_attr_dim:
            pad = torch.zeros(self.global_attr_dim - self.machine_attr_dim)
            attrs = torch.cat([attrs, pad], dim=0)

        out = {
            "patches_1": p1,
            "attrs":     attrs,
            "filename":  fname
        }

        # second view for contrastive train
        if "train" in path:
            v2 = self.transform(img)
            p2 = self.extract_patches(v2)
            if self.max_patches and p2.size(0) > self.max_patches:
                p2 = p2[sel]
            out["patches_2"] = p2

        return out

class ASTRA_EvalRGBDataset(ASTRA_AttnPatchRGBDataset):
    """
    Deterministic eval dataset.  Returns:
      - 'patches' : (N,3,224,224)
      - 'attrs'   : (global_attr_dim,)
      - 'label'   : 0 or 1
      - 'filename'
    """
    def __init__(self, *args, **kwargs):
        super()._init_(*args, split=kwargs.pop("split","test"), **kwargs)
        # override transform
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])

    def __getitem__(self, idx):
        base = super()._getitem_(idx)
        patches = base["patches_1"]
        label = 0 if "normal" in base["filename"] else 1
        return {
            "patches":  patches,
            "attrs":    base["attrs"],
            "label":    label,
            "filename": base["filename"]
        }

