# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Datamanager.
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from rich.progress import Console
    
CONSOLE = Console(width=120)

from dig.data.utils.dino_dataloader import DinoDataloader
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.cameras.cameras import Cameras

@dataclass
class DiGDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: DiGDataManager)
    
class DiGDataManager(FullImageDatamanager):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        
        dino_cache_path = Path(osp.join(cache_dir, "dino.npy"))
        images = [self.cached_train[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))]
        images = torch.cat(images)
        self.dino_dataloader = DinoDataloader(
            image_list = images,
            device = self.device,
            cfg={"image_shape": list(images.shape[2:4])},
            cache_path=dino_cache_path
        )

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_train(step)
        data["dino"] = self.dino_dataloader.get_full_img_feats(camera.metadata["cam_idx"])
        return camera, data
    