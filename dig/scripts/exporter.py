# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

import json
import os
import sys
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import collect_camera_poses, generate_point_cloud, get_mesh_from_filename
from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
# from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

import torch.nn.functional as F

def identity_7vec(device="cuda"):
    """
    Returns a 7-tensor of identity pose, as wxyz_xyz.
    """
    return torch.tensor([[1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=device)

def configure_from_clusters(init_means, group_labels: torch.Tensor):
    """
    Given `group_labels`, set the group masks and labels.

    Affects all attributes affected by # of parts:
    - `self.num_groups
    - `self.group_labels`
    - `self.group_masks`
    - `self.part_deltas`
    - `self.init_p2o`
    - `self.atap`
    , as well as `self.init_o2w`.

    NOTE(cmk) why do you need to store both `self.group_labels` and `self.group_masks`?
    """
    # Get group / cluster label info.
    
    num_groups = int(group_labels.max().item() + 1)
    group_masks = [(group_labels == cid).cuda() for cid in range(group_labels.max() + 1)]

    # Initialize the part poses to identity. Again, wxyz_xyz.
    # Parts are initialized at the centroid of the part cluster.
    init_p2o = identity_7vec().repeat(num_groups, 1)
    for i, g in enumerate(group_masks):
        gp_centroid = init_means[g].mean(dim=0)
        init_p2o[i, 4:] = gp_centroid - init_means.mean(dim=0)
    
    return num_groups, group_masks, init_p2o

@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path


def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check that the pipeline is valid for this exporter.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        metadata = {"directions_norm": torch.linalg.vector_norm(directions, dim=-1, keepdim=True)}
        ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            metadata=metadata,
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = False
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    # def main(self) -> None:
    #     """Export point cloud."""

    #     if not self.output_dir.exists():
    #         self.output_dir.mkdir(parents=True)

    #     _, pipeline, _, _ = eval_setup(self.load_config)

    #     validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

    #     # Increase the batchsize to speed up the evaluation.
    #     assert isinstance(
    #         pipeline.datamanager,
    #         (VanillaDataManager, ParallelDataManager),
    #     )
    #     assert pipeline.datamanager.train_pixel_sampler is not None
    #     pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

    #     # Whether the normals should be estimated based on the point cloud.
    #     estimate_normals = self.normal_method == "open3d"
    #     crop_obb = None
    #     if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
    #         crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
    #     pcd = generate_point_cloud(
    #         pipeline=pipeline,
    #         num_points=self.num_points,
    #         remove_outliers=self.remove_outliers,
    #         reorient_normals=self.reorient_normals,
    #         estimate_normals=estimate_normals,
    #         rgb_output_name=self.rgb_output_name,
    #         depth_output_name=self.depth_output_name,
    #         normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
    #         crop_obb=crop_obb,
    #         std_ratio=self.std_ratio,
    #     )
    #     if self.save_world_frame:
    #         # apply the inverse dataparser transform to the point cloud
    #         points = np.asarray(pcd.points)
    #         poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
    #         poses[:, :3, 3] = points
    #         poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
    #             torch.from_numpy(poses)
    #         )
    #         points = poses[:, :3, 3].numpy()
    #         pcd.points = o3d.utility.Vector3dVector(points)

    #     torch.cuda.empty_cache()

    #     CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
    #     CONSOLE.print("Saving Point Cloud...")
    #     tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    #     # The legacy PLY writer converts colors to UInt8,
    #     # let us do the same to save space.
    #     tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
    #     o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
    #     print("\033[A\033[A")
    #     CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    refine_mesh_using_initial_aabb_estimate: bool = False
    """Refine the mesh using the initial AABB estimate."""
    refinement_epsilon: float = 1e-2
    """Refinement epsilon for the mesh. This is the distance in meters that the refined AABB/OBB will be expanded by
    in each direction."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            refine_mesh_using_initial_aabb_estimate=self.refine_mesh_using_initial_aabb_estimate,
            refinement_epsilon=self.refinement_epsilon,
        )

        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportPoissonMesh(Exporter):
    """
    Export a mesh using poisson surface reconstruction.
    """

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_point_cloud: bool = False
    """Whether to save the point cloud."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (VanillaDataManager, ParallelDataManager),
        )
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        else:
            crop_obb = None

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()
        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")

        if self.save_point_cloud:
            CONSOLE.print("Saving Point Cloud...")
            o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

        CONSOLE.print("Computing Mesh... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

        CONSOLE.print("Saving Mesh...")
        o3d.io.write_triangle_mesh(str(self.output_dir / "poisson_mesh.ply"), mesh)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

        # This will texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the poisson reconstruction
            mesh = get_mesh_from_filename(
                str(self.output_dir / "poisson_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """Export a mesh using marching cubes."""

    isosurface_threshold: float = 0.0
    """The isosurface threshold for extraction. For SDF based methods the surface is the zero level set."""
    resolution: int = 1024
    """Marching cube resolution."""
    simplify_mesh: bool = False
    """Whether to simplify the mesh."""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Minimum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Maximum of the bounding box."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Main function."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # TODO: Make this work with Density Field
        assert hasattr(pipeline.model.config, "sdf_field"), "Model must have an SDF field."

        CONSOLE.print("Extracting mesh with marching cubes... which may take a while")

        assert self.resolution % 512 == 0, f"""resolution must be divisible by 512, got {self.resolution}.
        This is important because the algorithm uses a multi-resolution approach
        to evaluate the SDF where the minimum resolution is 512."""

        # Extract mesh using marching cubes for sdf at a multi-scale resolution.
        multi_res_mesh = generate_mesh_with_multires_marching_cubes(
            geometry_callable_field=lambda x: cast(SDFField, pipeline.model.field)
            .forward_geonetwork(x)[:, 0]
            .contiguous(),
            resolution=self.resolution,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            isosurface_threshold=self.isosurface_threshold,
            coarse_mask=None,
        )
        filename = self.output_dir / "sdf_marching_cubes_mesh.ply"
        multi_res_mesh.export(filename)

        # load the mesh from the marching cubes export
        mesh = get_mesh_from_filename(str(filename), target_num_faces=self.target_num_faces)
        CONSOLE.print("Texturing mesh with NeRF...")
        texture_utils.export_textured_mesh(
            mesh,
            pipeline,
            self.output_dir,
            px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )


@dataclass
class ExportCameraPoses(Exporter):
    """
    Export camera poses to a .json file.
    """

    def main(self) -> None:
        """Export camera poses"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        assert isinstance(pipeline, VanillaPipeline)
        train_frames, eval_frames = collect_camera_poses(pipeline)

        for file_name, frames in [("transforms_train.json", train_frames), ("transforms_eval.json", eval_frames)]:
            if len(frames) == 0:
                CONSOLE.print(f"[bold yellow]No frames found for {file_name}. Skipping.")
                continue

            output_file_path = os.path.join(self.output_dir, file_name)

            with open(output_file_path, "w", encoding="UTF-8") as f:
                json.dump(frames, f, indent=4)

            CONSOLE.print(f"[bold green]:white_check_mark: Saved poses to {output_file_path}")


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    output_filename: str = "point_cloud.ply"
    """Name of the output file."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    ply_color_mode: Literal["sh_coeffs", "rgb"] = "sh_coeffs"
    """If "rgb", export colors as red/green/blue fields. Otherwise, export colors as
    spherical harmonics coefficients."""

    @staticmethod
    def write_ply(
        filename: str,
        count: int,
        map_to_tensors: typing.OrderedDict[str, np.ndarray],
    ):
        """
        Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
        Note: All float values will be converted to float32 for writing.

        Parameters:
        filename (str): The name of the file to write.
        count (int): The number of vertices to write.
        map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
            Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
        """

        # Ensure count matches the length of all tensors
        if not all(tensor.size == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        # Type check for numpy arrays of type float or uint8 and non-empty
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

        with open(filename, "wb") as ply_file:
            nerfstudio_version = version("nerfstudio")
            # Write PLY header
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")
            ply_file.write(f"comment Generated by Nerstudio {nerfstudio_version}\n".encode())
            ply_file.write(b"comment Vertical Axis: z\n")
            ply_file.write(f"element vertex {count}\n".encode())

            # Write properties, in order due to OrderedDict
            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            # Write binary data
            # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())

    # def main(self) -> None:
    #     if not self.output_dir.exists():
    #         self.output_dir.mkdir(parents=True)

    #     _, pipeline, _, _ = eval_setup(self.load_config, test_mode="inference")
        
    #     pipeline.load_state() 
    #     pipeline.reset_colors()

    #     assert isinstance(pipeline.model, SplatfactoModel)

    #     model: SplatfactoModel = pipeline.model

    #     filename = self.output_dir / self.output_filename

    #     map_to_tensors = OrderedDict()
        
    #     init_means = model.gauss_params['means'].detach().clone()
    #     labels = pipeline.cluster_labels.int().cuda()
    #     # num_groups, group_masks, init_p2o = configure_from_clusters(init_means, labels)

    #     with torch.no_grad():
    #         positions = model.means.cpu().numpy()
    #         count = positions.shape[0]
    #         n = count
    #         map_to_tensors["x"] = positions[:, 0]
    #         map_to_tensors["y"] = positions[:, 1]
    #         map_to_tensors["z"] = positions[:, 2]
    #         map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
    #         map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
    #         map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

    #         if self.ply_color_mode == "rgb":
    #             colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
    #             colors = (colors * 255).astype(np.uint8)
    #             map_to_tensors["red"] = colors[:, 0]
    #             map_to_tensors["green"] = colors[:, 1]
    #             map_to_tensors["blue"] = colors[:, 2]
    #         elif self.ply_color_mode == "sh_coeffs":
    #             shs_0 = model.shs_0.contiguous().cpu().numpy()
    #             for i in range(shs_0.shape[1]):
    #                 map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

    #         if model.config.sh_degree > 0:
    #             if self.ply_color_mode == "rgb":
    #                 CONSOLE.print(
    #                     "Warning: model has higher level of spherical harmonics, ignoring them and only export rgb."
    #                 )
    #             elif self.ply_color_mode == "sh_coeffs":
    #                 # transpose(1, 2) was needed to match the sh order in Inria version
    #                 shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
    #                 shs_rest = shs_rest.reshape((n, -1))
    #                 for i in range(shs_rest.shape[-1]):
    #                     map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

    #         map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

    #         scales = model.scales.data.cpu().numpy()
    #         for i in range(3):
    #             map_to_tensors[f"scale_{i}"] = scales[:, i, None]

    #         quats = model.quats.data.cpu().numpy()
    #         for i in range(4):
    #             map_to_tensors[f"rot_{i}"] = quats[:, i, None]

    #         if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
    #             crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
    #             assert crop_obb is not None
    #             mask = crop_obb.within(torch.from_numpy(positions)).numpy()
    #             for k, t in map_to_tensors.items():
    #                 map_to_tensors[k] = map_to_tensors[k][mask]

    #             n = map_to_tensors["x"].shape[0]
    #             count = n
        
    #     if False:
    #         # apply the inverse dataparser transform to the point cloud
    #         filtered_position = np.concatenate([map_to_tensors["x"][:, None], map_to_tensors["y"][:, None], map_to_tensors["z"][:, None]], axis=1)
    #         filtered_quats = np.concatenate([map_to_tensors[f"rot_{i}"][:, None] for i in range(4)], axis=1).squeeze(axis=-1)
            
    #         points = np.zeros((filtered_position.shape[0],3,4))
    #         points[:,:3,3] = filtered_position[:,:3]
            
    #         points[:,:3,:3] = quat_to_rotmat(torch.from_numpy(filtered_quats)).numpy()
            
    #         poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
    #         poses[:, :3, :] = points
    #         poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
    #             torch.from_numpy(poses)
    #         )
            
    #         print(pipeline.datamanager.train_dataparser_outputs.dataparser_transform)
    #         print(pipeline.datamanager.train_dataparser_outputs.dataparser_scale)
    #         for i in range(3):
    #             map_to_tensors[f"scale_{i}"] = scales[:, i]
            
    #         xyz = poses[:, :3, 3]
    #         rot = poses[:, :3, :3]
    #         quat = matrix_to_quaternion(rot)
    #         map_to_tensors["x"] = xyz[:, 0].numpy()
    #         map_to_tensors["y"] = xyz[:, 1].numpy()
    #         map_to_tensors["z"] = xyz[:, 2].numpy()
    #         for qi in range(4):
    #             map_to_tensors[f"rot_{qi}"] = quat[:, qi, None].numpy()
            
    #         # for i in range(3):
    #         #     map_to_tensors[f"scale_{i}"] = np.exp(scales[:, i, None])
    #         #     map_to_tensors["opacity"] = torch.sigmoid(model.opacities.data).cpu().numpy()
        
    #     # post optimization, it is possible have NaN/Inf values in some attributes
    #     # to ensure the exported ply file has finite values, we enforce finite filters.
    #     select = np.ones(n, dtype=bool)
    #     for k, t in map_to_tensors.items():
    #         n_before = np.sum(select)
    #         select = np.logical_and(select, np.isfinite(t).all(axis=-1))
    #         n_after = np.sum(select)
    #         if n_after < n_before:
    #             CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")
    #     nan_count = np.sum(select) - n

    #     # filter gaussians that have opacities < 1/255, because they are skipped in cuda rasterization
    #     low_opacity_gaussians = (map_to_tensors["opacity"]).squeeze(axis=-1) < -5.5373  # logit(1/255)
        
    #     lowopa_count = np.sum(low_opacity_gaussians)
    #     select[low_opacity_gaussians] = 0

    #     if np.sum(select) < n:
    #         CONSOLE.print(
    #             f"{nan_count} Gaussians have NaN/Inf and {lowopa_count} have low opacity, only export {np.sum(select)}/{n}"
    #         )
    #         for k, t in map_to_tensors.items():
    #             map_to_tensors[k] = map_to_tensors[k][select]
    #         count = np.sum(select)
            

    #     ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
    ]
]

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def quat_to_rotmat(quat):
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))



# def render_part_view(
#     output_dir,
#     dig_config_path: Path,
#     state_path: str,
#     joint_origin: list = None,
# ):
#     v_train_config,pipeline,_,_ = eval_setup(dig_config_path)
    
#     v = Viewer(ViewerConfig(default_composite_depth=False,num_rays_per_chunk=-1),dig_config_path.parent,pipeline.datamanager.get_datapath(),pipeline,train_lock=Lock())
#     v_train_config.logging.local_writer.enable = False
#     writer.setup_local_writer(v_train_config.logging, max_iter = v_train_config.max_num_iterations)
    
#     pipeline.load_state_from_path(state_path_filename = state_path) 
#     pipeline.reset_colors()
#     pipeline.model.eval()
#     init_means = pipeline.model.gauss_params['means'].detach().clone()
#     labels = pipeline.cluster_labels.int().cuda()
#     num_groups, group_masks, init_p2o = configure_from_clusters(init_means, labels)
#     if joint_origin is None:
#         joint_origin = {}
#         for idx in range(num_groups):
#             joint_origin[idx] = np.zeros(3)
        
#     nerfcameras = deepcopy(pipeline.datamanager.train_dataset.cameras)
#     image_names = pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames
#     # remove distortion params 
#     for c in nerfcameras:
#         c.distortion_params[:] = torch.zeros_like(c.distortion_params)
        
#     os.makedirs(os.path.join(output_dir, f"all"), exist_ok=True)
#     with torch.no_grad():
#         for idx in tqdm(range(len(nerfcameras))):
#             outputs = pipeline.model.get_outputs(nerfcameras[idx:idx+1].to('cuda'))
#             height, width = nerfcameras[idx:idx+1].height, nerfcameras[idx:idx+1].width
#             img_path = image_names[idx]
            
#             forground_mask = outputs["accumulation"] > 0.8
#             forground_mask = forground_mask[:, :, 0].detach().cpu().numpy()
#             forground_mask = Image.fromarray((forground_mask*255).astype(np.uint8))
#             forground_mask.save(os.path.join(output_dir, f"all", f"{img_path.stem}.jpg.png"))
            
def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa