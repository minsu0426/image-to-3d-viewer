from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# torchmcubes가 없으면 기본 trimesh 모듈을 쓰도록 우회 세팅
try:
    import torchmcubes
    HAS_TORCHMCUBES = True
except ImportError:
    import trimesh
    HAS_TORCHMCUBES = False


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        
        if HAS_TORCHMCUBES:
            try:
                v_pos, t_pos_idx = torchmcubes.marching_cubes(level.detach(), 0.0)
            except AttributeError:
                print("torchmcubes was not compiled with CUDA support, use CPU version instead.")
                v_pos, t_pos_idx = torchmcubes.marching_cubes(level.detach().cpu(), 0.0)
        else:
            # 파이썬 3.13 우회 연산 (trimesh 사용)
            level_np = level.detach().cpu().numpy()
            verts, faces = trimesh.isosurface.marching_cubes(level_np, level=0.0)
            
            # 파이토치 텐서로 재변환
            v_pos = torch.from_numpy(verts.astype(np.float32)).to(level.device)
            t_pos_idx = torch.from_numpy(faces.astype(np.int64)).to(level.device)

        v_pos = v_pos[..., [2, 1, 0]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)