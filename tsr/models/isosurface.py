from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torchmcubes import marching_cubes as _torchmcubes_marching_cubes
    TORCHMCUBES_AVAILABLE = True
except ImportError:
    TORCHMCUBES_AVAILABLE = False


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

    def forward(self, x: torch.Tensor):
        # system.py 호출: self.isosurface_helper(-(density - threshold))
        # x shape: (R*R*R, 1) → (R, R, R) 으로 reshape 필요

        if TORCHMCUBES_AVAILABLE:
            # torchmcubes는 (R,R,R) float32 필요
            x_3d = x.reshape(self.resolution, self.resolution, self.resolution).float()
            vertices, faces = _torchmcubes_marching_cubes(x_3d, 0.0)
            vertices = vertices.to(x.device)
            faces = faces.to(x.device)
        else:
            from skimage import measure

            # (R*R*R, 1) → (R, R, R) reshape
            x_3d = x.reshape(self.resolution, self.resolution, self.resolution)
            x_np = x_3d.detach().cpu().float().numpy()

            x_min, x_max = float(x_np.min()), float(x_np.max())
            print(f"[isosurface] shape={x_np.shape} min={x_min:.4f} max={x_max:.4f}")

            if x_min >= 0.0:
                level = float(np.percentile(x_np, 10))
            elif x_max <= 0.0:
                level = float(np.percentile(x_np, 90))
            else:
                level = 0.0

            try:
                verts, faces_np, normals, values = measure.marching_cubes(
                    x_np, level=level
                )
                print(f"[isosurface] SUCCESS verts={len(verts)} faces={len(faces_np)}")
            except (ValueError, RuntimeError) as e:
                print(f"[isosurface] FAILED: {e}")
                vertices = torch.zeros((0, 3), dtype=torch.float32, device=x.device)
                faces = torch.zeros((0, 3), dtype=torch.long, device=x.device)
                return vertices, faces

            res = x_np.shape[0]
            verts = verts / max(res - 1, 1)
            vertices = torch.tensor(verts, dtype=torch.float32, device=x.device)
            faces = torch.tensor(faces_np.astype(np.int64), dtype=torch.long, device=x.device)

        return vertices, faces