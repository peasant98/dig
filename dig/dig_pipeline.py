import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Mapping, Any, Optional, List, Dict
from torchtyping import TensorType
from pathlib import Path
import trimesh
import viser
import viser.transforms as vtf
import open3d as o3d
import cv2
import time

import torch
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from torch.cuda.amp.grad_scaler import GradScaler
from nerfstudio.viewer.viewer_elements import *
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.models.splatfacto import SplatfactoModel

from cuml.cluster.hdbscan import HDBSCAN
from nerfstudio.models.splatfacto import RGB2SH, SH2RGB

import tqdm

from sklearn.neighbors import NearestNeighbors
from enum import Enum, auto

from garfield.garfield_datamanager import GarfieldDataManagerConfig, GarfieldDataManager
from garfield.garfield_model import GarfieldModel, GarfieldModelConfig
from garfield.garfield_pipeline import GarfieldPipelineConfig, GarfieldPipeline

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

def generate_random_colors(N=5000) -> torch.Tensor:
    """Generate random colors for visualization"""
    hs = np.random.uniform(0, 1, size=(N, 1))
    ss = np.random.uniform(0.6, 0.61, size=(N, 1))
    vs = np.random.uniform(0.84, 0.95, size=(N, 1))
    hsv = np.concatenate([hs, ss, vs], axis=-1)
    # convert to rgb
    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8)[None, ...], cv2.COLOR_HSV2RGB)
    return torch.Tensor(rgb.squeeze() / 255.0)

class ObjectMode(Enum):
    ARTICULATED = auto()
    RIGID_OBJECTS = auto()

@dataclass
class DigPipelineConfig(VanillaPipelineConfig):
    """Gaussian Splatting, but also loading GARField grouping field from ckpt."""
    _target: Type = field(default_factory=lambda: DiGPipeline)
    garfield_ckpt: Optional[Path] = None  # Need to specify config.yml
    filter_table_plane: Optional[bool] = True


class DiGPipeline(VanillaPipeline):
    """
    Trains a Gaussian Splatting model, but also loads a GARField grouping field from ckpt.
    This grouping field allows you to:
     - interactive click-based group selection (you can drag it around)
     - scene clustering, then group selection (also can drag it around)

    Note that the pipeline training must be stopped before you can interact with the scene!!
    """
    model: SplatfactoModel
    garfield_pipeline: List[GarfieldPipeline]  # To avoid importing Viewer* from nerf pipeline
    state_stack: List[Dict[str, TensorType]]  # To revert to previous state
    click_location: Optional[TensorType]  # For storing click location
    click_handle: Optional[viser.GlbHandle]  # For storing click handle
    crop_group_list: List[TensorType]  # For storing gaussian crops (based on click point)
    crop_transform_handle: Optional[viser.TransformControlsHandle]  # For storing scene transform handle -- drag!
    cluster_labels: Optional[TensorType]  # For storing cluster labels

    def __init__(
        self,
        config: DigPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: typing.Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)

        print("Loading instance feature model...")
        if config.garfield_ckpt is None:
            print("\n\n GARField checkpoint not specified, continuing without segmentation capabilities. \
                  To run the full 4D-DPM pipeline you'll need to train GARField first and provide the trained checkpoint file as --pipeline.garfield-ckpt /path/to/config.yml \n\n")
            dec = input("Continue? [y]/n")
            if 'n' in dec.lower():
                exit()
        else:
            from nerfstudio.utils.eval_utils import eval_setup
            _, garfield_pipeline, _, _ = eval_setup(
                config.garfield_ckpt, test_mode="inference"
            )
            self.garfield_pipeline = [garfield_pipeline]
        if self.has_garfield:
            self.state_stack = []

            self.colormap = generate_random_colors()

            self.viewer_control = ViewerControl()
            
            self.num_clusters = ViewerNumber(name="Num. Clusters", default_value=0, disabled = True, visible=False)
            
            self.a_segment_done_button = ViewerButtonGroup(name="Selection Type", cb_hook=self._update_interaction_method, default_value = 'Click',options=['Click','Cluster'])
            

            self.click_gaussian = ViewerButton(name="Click", cb_hook=self._click_gaussian)
            self.last_two_clicks = []
            
            self.click_location = None
            self.click_handle = None

            self.crop_to_click = ViewerButton(name="Crop to Click", cb_hook=self._crop_to_click, disabled=True)
            self.crop_to_group_level = ViewerSlider(name="Group Level", min_value=0, max_value=29, step=1, default_value=0, cb_hook=self._update_crop_vis, disabled=True)
            self.crop_group_list = []

            self.cluster_scene_z = ViewerButton(name="Cluster Scene", cb_hook=self._cluster_scene, disabled=False, visible=False)
            self.cluster_scene_scale = ViewerSlider(name="Cluster Scale", min_value=0.0, max_value=2.0, step=0.01, default_value=0.0, disabled=False, visible=False)
            self.z_cluster_scene_shuffle_colors = ViewerButton(name="Reshuffle Cluster Colors", cb_hook=self._reshuffle_cluster_colors, disabled=False, visible=False)
            self.cluster_labels = None

            self.click_save_state_rigid = ViewerButton(name="Save Rigid-Body State", cb_hook=self._save_state_rigid, disabled=True, )
            
            self.d_reset_state = ViewerButton(name="Undo", cb_hook=self._reset_state, disabled=True)

            dataset_name = config.datamanager.data.stem
            save_state_dir = Path(f"outputs/{dataset_name}")
            save_state_dir.mkdir(parents=True, exist_ok=True)
            self.state_dir = save_state_dir
            self.state_file = save_state_dir / "state.pt"
            self.load_button = ViewerButton("Load State", cb_hook=lambda _:self.load_state(), visible=True)

            self.combine_clusters_button = ViewerButton("Combine Last 2 Clicked Clusters", cb_hook=self._combine_last_two_clicks, visible=True, disabled=True)
            
    @property
    def has_garfield(self):
        return 'garfield_pipeline' in self.__dict__

    def save_state(self, _):
        """Save the current state of the model."""
        print("Saving state...")
        #save the current state
        params_to_save = {}
        for k, v in self.model.gauss_params.items():
            params_to_save[k] = v  # assuming v is already a torch.Tensor

        #save label
        params_to_save["cluster_labels"] = self.cluster_labels

        torch.save(params_to_save, self.state_file)
        print(f"State saved to {self.state_file}")
        
    def _save_state_rigid(self, button: ViewerButton):
        from datetime import datetime
        """Save the current state of the model."""
        #save the current state
        params_to_save = {}
        for k, v in self.model.gauss_params.items():
            params_to_save[k] = v  # assuming v is already a torch.Tensor
        
        pathlibstr = Path(f"state_rigid"+(datetime.now().strftime("_%Y%m%d_%H%M%S")+".pt"))
        datetime_state_file = self.state_dir / pathlibstr
        
        # debug img
        # image_file = self.state_dir / pathlibstr.with_suffix(".png")
        # cam = self.viewer_control.get_camera(540, 960).to(self.device)
        # output = self.model.get_outputs(cam)
        # cv2.imwrite(str(image_file), cv2.cvtColor((output['rgb'].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        torch.save(params_to_save, datetime_state_file)
        print(f"State saved to {datetime_state_file}")
        self.click_save_state_rigid.set_disabled(True)
        
    def load_state(self):
        # For now we only handle either multi-rigid or articulated body, not yet both
        # If an articulated state.pt exists, that one gets loaded and none of the rigid states are loaded
        rigid_state_files = list(self.state_dir.glob("state_rigid_*.pt"))
        self.fixed_obj_ids = []
        if self.state_file.exists(): # Load articulated state and clusters from state.pt if it exists
            self._queue_state()
            loaded_state = torch.load(self.state_file)
            for name in self.model.gauss_params.keys():
                self.model.gauss_params[name] = loaded_state[name].clone().to(self.device)
            self.cluster_labels = loaded_state["cluster_labels"]
            self.model.cluster_labels = self.cluster_labels.to(self.device)
            
            self.object_mode = ObjectMode.ARTICULATED
            
        elif len(rigid_state_files) > 0:
            # Load rigid state multiple state files
            self._queue_state()
            params = {}
            len_params = []
            for idx, state_file in enumerate(rigid_state_files):
                loaded_state = torch.load(state_file)
                for name in self.model.gauss_params.keys():
                    if name not in params:
                        params[name] = []  # Initialize empty list for new keys
                    if name not in loaded_state.keys():
                        print(f"Warning: {name} not found in {state_file}, skipping.")
                        continue
                    params[name].append(loaded_state[name].clone().to(self.device))
                len_params.append(len(loaded_state['means']))
                
                # Check if fixed is in the name of the file, in which case we store info about fixed masks (for turning off certain tracking features on this object)
                if 'fixed' in state_file.name:
                    self.fixed_obj_ids.append(idx)
                
            for name in self.model.gauss_params.keys():
                params[name] = torch.cat(params[name], dim=0)
                self.model.gauss_params[name] = params[name].clone().to(self.device)
            self.cluster_labels = torch.cat([torch.full((length,), i, dtype=torch.float) for i, length in enumerate(len_params)])
            self.model.cluster_labels = self.cluster_labels.to(self.device)
            
            self.object_mode = ObjectMode.RIGID_OBJECTS
        else:
            print(f"No state file found at {self.state_file} or in {self.state_dir}, unable to load state.")
            return
        
        if self.num_clusters.gui_handle is not None:
            self.num_clusters.set_hidden(False)
            self.num_clusters.gui_handle.value = len(self.cluster_labels.unique())

    
    def reset_colors(self):
        from cuml.neighbors import NearestNeighbors
        model = NearestNeighbors(n_neighbors=1)
        original_means = self.state_stack[0]["means"].detach().cpu().numpy()
        original_features_dc = self.state_stack[0]["features_dc"].detach().clone()
        original_features_rest = self.state_stack[0]["features_rest"].detach().clone()
        model.fit(original_means)
        curmeans = self.model.means.detach().cpu().numpy()
        _, match_ids = model.kneighbors(curmeans)
        self.model.gauss_params["features_dc"] = original_features_dc[match_ids.squeeze()].cuda()
        self.model.gauss_params["features_rest"] = original_features_rest[match_ids.squeeze()].cuda()
        match_ids = torch.tensor(match_ids,dtype=torch.long,device='cuda')
    
    def _update_interaction_method(self, handle):
        """Update the UI based on the interaction method"""
        hide_in_interactive = (not (handle.value == "Click")) # i.e., hide if in interactive mode

        self.cluster_scene_z.set_hidden((not hide_in_interactive))
        self.cluster_scene_scale.set_hidden((not hide_in_interactive))
        self.z_cluster_scene_shuffle_colors.set_hidden((not hide_in_interactive))

        self.click_gaussian.set_hidden(hide_in_interactive)
        self.crop_to_click.set_hidden(hide_in_interactive)
        self.combine_clusters_button.set_hidden(hide_in_interactive)
        self.click_save_state_rigid.set_hidden(hide_in_interactive)
        self.crop_to_group_level.set_hidden(hide_in_interactive)

    def _reset_state(self, button: ViewerButton):
        """Revert to previous saved state"""
        assert len(self.state_stack) > 0, "No previous state to revert to"
        prev_state = self.state_stack.pop()
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name]

        self.click_location = None
        if self.click_handle is not None:
            self.click_handle.remove()
        self.click_handle = None

        self.click_gaussian.set_disabled(False)

        self.crop_to_click.set_disabled(True)
        self.crop_to_group_level.set_disabled(True)
        # self.crop_to_group_level.value = 0
        self.crop_group_list = []
        if len(self.state_stack) == 0:
            self.d_reset_state.set_disabled(True)

        self.cluster_labels = None
        self.cluster_scene_z.set_disabled(False)
        self.click_save_state_rigid.set_disabled(True)

    def _queue_state(self):
        """Save current state to stack"""
        import copy
        self.state_stack.append(copy.deepcopy({k:v.detach() for k,v in self.model.gauss_params.items()}))
        if self.d_reset_state.gui_handle is not None:
            self.d_reset_state.set_disabled(False)

    def _click_gaussian(self, button: ViewerButton):
        """Start listening for click-based 3D point specification.
        Refer to garfield_interaction.py for more details."""
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)
            self.click_gaussian.set_disabled(False)
            self.crop_to_click.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)

        self.click_gaussian.set_disabled(True)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)

    def _on_rayclick(self, click: ViewerClick):
        """On click, calculate the 3D position of the click and visualize it.
        Refer to garfield_interaction.py for more details."""

        cam = self.viewer_control.get_camera(500, None, 0)
        cam2world = cam.camera_to_worlds[0, :3, :3]
        import viser.transforms as vtf

        x_pi = vtf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)
        world2cam = (cam2world @ x_pi).inverse()
        # rotate the ray around into cam coordinates
        newdir = world2cam @ torch.tensor(click.direction).unsqueeze(-1)
        z_dir = newdir[2].item()
        # project it into coordinates with matrix
        K = cam.get_intrinsics_matrices()[0]
        coords = K @ newdir
        coords = coords / coords[2]
        pix_x, pix_y = int(coords[0]), int(coords[1])
        self.model.eval()
        outputs = self.model.get_outputs(cam.to(self.device))
        self.model.train()
        with torch.no_grad():
            depth = outputs["depth"][pix_y, pix_x].cpu().numpy()

        self.click_location = np.array(click.origin) + np.array(click.direction) * (depth / z_dir)

        sphere_mesh = trimesh.creation.icosphere(radius=0.2)
        sphere_mesh.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)  # type: ignore
        self.click_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/click",
            mesh=sphere_mesh,
            position=VISER_NERFSTUDIO_SCALE_RATIO * self.click_location,
        )
        
        # Add clicked cluster to 'last_two_clicks' history if cluster labels are available
        if self.cluster_labels is not None:
            # get the closest 10 points to the sphere, using kdtree
            ind = self._click_location_to_gauss_inds(10) 
            # find most represented cluster
            clicked_cluster = self.cluster_labels[list(ind)].to(torch.int).mode().values.item()
            
            # Check if this cluster was just clicked
            if len(self.last_two_clicks) > 0 and clicked_cluster == self.last_two_clicks[-1]:
                print(f"Cluster {clicked_cluster} was already clicked. Click a different cluster to combine.")
                return
                
            self.last_two_clicks.append(clicked_cluster)
            if len(self.last_two_clicks) > 2:
                self.last_two_clicks.pop(0)
            if len(self.last_two_clicks) == 2:
                self.combine_clusters_button.set_disabled(False)
    
    def _combine_last_two_clicks(self, button: ViewerButton):
        if len(self.last_two_clicks) == 2:
            self._queue_state()
            unique_labels = self.cluster_labels.unique().sort()[0]
        
            # Determine which label is larger (we'll map this one to the smaller one)
            source_label = max(self.last_two_clicks[0], self.last_two_clicks[1])
            target_label = min(self.last_two_clicks[0], self.last_two_clicks[1])
            
            # Combine the clusters
            self.cluster_labels[self.cluster_labels == source_label] = target_label
            
            # Shift down all labels above the source_label by 1 to maintain consecutive labels
            for label in unique_labels:
                if label > source_label:
                    self.cluster_labels[self.cluster_labels == label] -= 1
            self.viewer_control.viewer._trigger_rerender()
            
            self.last_two_clicks = []
            self.combine_clusters_button.set_disabled(True)
            self.num_clusters.gui_handle.value = len(self.cluster_labels.unique())
            self._reshuffle_cluster_colors(None)
            self.save_state(None)
        else:
            print("Need to have two clicks in history to combine clusters.")
            return

    def _click_location_to_gauss_inds(self, top_k=10):
        """Get the closest gaussians to the click location"""
        curr_means = self.model.gauss_params['means'].detach()
        self.model.eval()

        # Get the 3D location of the click
        location = self.click_location
        location = torch.tensor(location).view(1, 3).to(self.device)

        # Create a kdtree, to get the closest gaussian to the click-point.
        points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(curr_means.cpu().numpy()))
        kdtree = o3d.geometry.KDTreeFlann(points)
        _, inds, _ = kdtree.search_knn_vector_3d(location.view(3, -1).float().detach().cpu().numpy(), top_k)

        # get the closest point to the sphere, using kdtree
        return inds
        
    def _estimate_table_plane(self, means):
        points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(means.cpu().numpy()))
        # First segment the plane
        plane_model, inliers = points.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane_model

        # Convert plane normal to unit vector
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        # Check if plane normal is roughly aligned with Z axis
        z_axis = np.array([0, 0, 1])
        # Check both upward and downward facing normals
        dot_product = max(np.abs(np.dot(normal, z_axis)), np.abs(np.dot(-normal, z_axis)))
        if dot_product < 0.7:  # cos(45 degrees) ≈ 0.707
            print("Warning: Detected plane is not horizontal")
            return None
        if np.abs(np.dot(normal, z_axis)) < -0.7:
            normal = -normal
        
        # Calculate signed distances from all points to the plane
        points_np = np.asarray(points.points)
        signed_dists = (points_np @ normal + d) / np.linalg.norm(normal)

        # Points above the plane have positive distance, below have negative
        above_mask = signed_dists > 0.01  # Small threshold to avoid numerical issues
        below_mask = signed_dists < -0.01

        # Convert to tensor indices
        above_indices = torch.from_numpy(np.where(above_mask)[0])
        below_indices = torch.from_numpy(np.where(below_mask)[0])

        return {
            'above': above_indices,
            'below': below_indices,
            'plane_model': plane_model
        }
        
    def _crop_to_click(self, button: ViewerButton):
        """Crop to click location"""
        assert self.click_location is not None, "Need to specify click location"

        self._queue_state()  # Save current state
        curr_means = self.model.gauss_params['means'].detach()
        # curr_dcsh = self.model.gauss_params['features_dc'].detach()
        # curr_rgb = SH2RGB(curr_dcsh)
        self.model.eval()
        
        plane_out = self._estimate_table_plane(curr_means)
        # server2 = viser.ViserServer()
        # server2.scene.add_point_cloud(
        #     name='/above_points',
        #     points=curr_means[out['above']].cpu().detach().numpy(),
        #     colors=curr_rgb[out['above']].cpu().detach().numpy(),
        #     point_size=0.005,
        #     point_shape='circle'
        # )
        # server2.scene.add_point_cloud(
        #     name='/below_points',
        #     points=curr_means[out['below']].cpu().detach().numpy(),
        #     colors=curr_rgb[out['below']].cpu().detach().numpy(),
        #     point_size=0.005,
        #     point_shape='circle'
            
        # )
        # import pdb ; pdb.set_trace()
        # The only way to reset is to reset the state using the reset button.
        self.click_gaussian.set_disabled(True)  # Disable user from changing click
        self.crop_to_click.set_disabled(True)  # Disable user from changing click

        # Get the 3D location of the click
        location = self.click_location
        location = torch.tensor(location).view(1, 3).to(self.device)

        # The list of positions to query for garfield features. The first one is the click location.
        positions = torch.cat([location, curr_means])  # N x 3

        # Create a kdtree, to get the closest gaussian to the click-point.
        points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(curr_means.cpu().numpy()))
        kdtree = o3d.geometry.KDTreeFlann(points)
        _, inds, _ = kdtree.search_knn_vector_3d(location.view(3, -1).float().detach().cpu().numpy(), 10)

        # get the closest point to the sphere, using kdtree
        sphere_inds = inds
        # scales = torch.ones((positions.shape[0], 1)).to(self.device)

        keep_list = []
        prev_group = None

        # Iterate over different scales, to get the a range of possible groupings.
        grouping_model = self.garfield_pipeline[0].model
        for s in tqdm.tqdm(torch.linspace(0, 1.5, 30)):
            # Calculate the grouping features, and calculate the affinity between click point and scene
            instances = grouping_model.get_grouping_at_points(positions, s)  # (1+N, 256)
            click_instance = instances[0]
            affinity = torch.norm(click_instance - instances, dim=1)[1:]

            # Filter out points that have affinity < 0.5 (i.e., not likely to be in the same group)
            keeps = torch.where(affinity < 0.5)[0].cpu()
            if self.config.filter_table_plane and plane_out is not None:
                # Filter out points that are above the table plane
                # Get indices that are in both keeps and above_plane using intersection
                keeps = keeps[torch.isin(keeps, plane_out['above'])]
            keep_points = points.select_by_index(keeps.tolist())  # indices of gaussians

            # Here, we desire the gaussian groups to be grouped tightly together spatially. 
            # We use DBSCAN to group the gaussians together, and choose the cluster that contains the click point.
            # Note that there may be spuriously high affinity between points that are spatially far apart,
            #  possibly due two different groups being considered together at an odd angle / far viewpoint.

            # If there are too many points, we downsample them first before DBSCAN.
            # Then, we assign the filtered points to the cluster of the nearest downsampled point.
            if len(keeps) > 5000:
                curr_point_min = keep_points.get_min_bound()
                curr_point_max = keep_points.get_max_bound()

                downsample_size = 0.01 * s
                _, _, curr_points_ds_ids = keep_points.voxel_down_sample_and_trace(
                    voxel_size=max(downsample_size, 0.0001),
                    min_bound=curr_point_min,
                    max_bound=curr_point_max,
                )

                if len(curr_points_ds_ids) == keeps.shape[0]:
                    clusters = np.asarray(keep_points.cluster_dbscan(eps=0.02, min_points=5))
                
                else:
                    curr_points_ds_ids = np.array([points[0] for points in curr_points_ds_ids])
                    curr_points_ds = keep_points.select_by_index(curr_points_ds_ids)
                    curr_points_ds_selected = np.zeros(len(keep_points.points), dtype=bool)
                    curr_points_ds_selected[curr_points_ds_ids] = True

                    _clusters = np.asarray(curr_points_ds.cluster_dbscan(eps=0.02, min_points=5))
                    nn_model = NearestNeighbors(
                        n_neighbors=1, algorithm="auto", metric="euclidean"
                    ).fit(np.asarray(curr_points_ds.points))


                    try:
                        _, indices = nn_model.kneighbors(np.asarray(keep_points.points)[~curr_points_ds_selected])
                    except:
                        import pdb; pdb.set_trace()

                    clusters = np.zeros(len(keep_points.points), dtype=int)
                    clusters[curr_points_ds_selected] = _clusters
                    clusters[~curr_points_ds_selected] = _clusters[indices[:, 0]]

            else:
                clusters = np.asarray(keep_points.cluster_dbscan(eps=0.02, min_points=5))

            # Choose the cluster that contains the click point. If there is none, move to the next scale.
            cluster_inds = clusters[np.isin(keeps, sphere_inds)]
            cluster_inds = cluster_inds[cluster_inds != -1]
            if len(cluster_inds) == 0:
                continue
            cluster_ind = cluster_inds[0]

            keeps = keeps[np.where(clusters == cluster_ind)]

            if prev_group is None:
                prev_group = keeps
                keep_list.append(keeps)
                continue

            keeps = torch.cat([prev_group, keeps])
            keeps = torch.unique(keeps)

            # # Deduplication, based on the # of current points included in the previous group.
            # overlap = torch.isin(keeps, prev_group).sum()
            # if overlap < 0.8 * len(keeps):
            #     prev_group = keeps
            keep_list.append(keeps)

        if len(keep_list) == 0:
            print("No gaussians within crop, aborting")
            # The only way to reset is to reset the state using the reset button.
            self.click_gaussian.set_disabled(False)
            self.crop_to_click.set_disabled(False)
            return

        # Remove the click handle + visualization
        self.click_location = None
        self.click_handle.remove()
        self.click_handle = None
        
        self.crop_group_list = keep_list
        self.crop_to_group_level.set_disabled(False)
        self.click_save_state_rigid.set_disabled(False)
        self.crop_to_group_level.value = 29

    def _update_crop_vis(self, number: ViewerSlider):
        """Update which click-based crop to visualize -- this requires that _crop_to_click has been called."""
        # If there is no click-based crop or saved state to crop from, do nothing
        if len(self.crop_group_list) == 0:
            return
        if len(self.state_stack) == 0:
            return
        
        # Clamp the number to be within the range of possible crops
        if number.value > len(self.crop_group_list) - 1:
            number.value = len(self.crop_group_list) - 1
            return
        elif number.value < 0:
            number.value = 0
            return

        keep_inds = self.crop_group_list[number.value]
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name][keep_inds]
        self.viewer_control.viewer._trigger_rerender()  # trigger viewer rerender

    def _reshuffle_cluster_colors(self, button: ViewerButton):
        """Reshuffle the cluster colors, if clusters defined using `_cluster_scene`."""
        if self.cluster_labels is None:
            return
        self.z_cluster_scene_shuffle_colors.set_disabled(True)  # Disable user from reshuffling colors
        self.colormap = generate_random_colors()
        colormap = self.colormap

        labels = self.cluster_labels

        features_dc = self.model.gauss_params['features_dc'].detach()
        features_rest = self.model.gauss_params['features_rest'].detach()
        for c_id in range(0, labels.max().int().item() + 1):
            # set the colors of the gaussians accordingly using colormap from matplotlib
            cluster_mask = np.where(labels == c_id)
            features_dc[cluster_mask] = RGB2SH(colormap[c_id, :3].to(self.model.gauss_params['features_dc']))
            features_rest[cluster_mask] = 0

        self.model.gauss_params['features_dc'] = torch.nn.Parameter(self.model.gauss_params['features_dc'])
        self.model.gauss_params['features_rest'] = torch.nn.Parameter(self.model.gauss_params['features_rest'])
        self.z_cluster_scene_shuffle_colors.set_disabled(False)

    def _cluster_scene(self, button: ViewerButton):
        """Cluster the scene, and assign gaussian colors based on the clusters.
        Also populates self.crop_group_list with the clusters group indices."""

        self._queue_state()  # Save current state
        self.cluster_scene_z.set_disabled(True)  # Disable user from clustering, while clustering

        scale = self.cluster_scene_scale.value
        grouping_model = self.garfield_pipeline[0].model
        
        positions = self.model.gauss_params['means'].detach()
        group_feats = grouping_model.get_grouping_at_points(positions, scale).cpu().numpy()  # (N, 256)
        positions = positions.cpu().numpy()

        start = time.time()

        # Cluster the gaussians using HDBSCAN.
        # We will first cluster the downsampled gaussians, then 
        #  assign the full gaussians to the spatially closest downsampled gaussian.

        vec_o3d = o3d.utility.Vector3dVector(positions)
        pc_o3d = o3d.geometry.PointCloud(vec_o3d)
        min_bound = np.clip(pc_o3d.get_min_bound(), -1, 1)
        max_bound = np.clip(pc_o3d.get_max_bound(), -1, 1)
        # downsample size to be a percent of the bounding box extent
        downsample_size = 0.007 * scale
        pc, _, ids = pc_o3d.voxel_down_sample_and_trace(
            max(downsample_size, 0.0001), min_bound, max_bound
        )
        if len(ids) > 1e6:
            print(f"Too many points ({len(ids)}) to cluster... aborting.")
            print( "Consider using interactive select to reduce points before clustering.")
            print( "Are you sure you want to cluster? Press y to continue, else return.")
            # wait for input to continue, if yes then continue, else return
            if input() != "y":
                self.cluster_scene_z.set_disabled(False)
                return

        id_vec = np.array([points[0] for points in ids])  # indices of gaussians kept after downsampling
        group_feats_downsampled = group_feats[id_vec]
        positions_downsampled = np.array(pc.points)

        print(f"Clustering {group_feats_downsampled.shape[0]} gaussians... ", end="", flush=True)

        # Run cuml-based HDBSCAN
        clusterer = HDBSCAN(
            cluster_selection_epsilon=0.1,
            min_samples=30,
            min_cluster_size=30,
            allow_single_cluster=True,
        ).fit(group_feats_downsampled)

        non_clustered = np.ones(positions.shape[0], dtype=bool)
        non_clustered[id_vec] = False
        labels = clusterer.labels_.copy()
        clusterer.labels_ = -np.ones(positions.shape[0], dtype=np.int32)
        clusterer.labels_[id_vec] = labels

        # Assign the full gaussians to the spatially closest downsampled gaussian, with scipy NearestNeighbors.
        positions_np = positions[non_clustered]
        if positions_np.shape[0] > 0:  # i.e., if there were points removed during downsampling
            k = 1
            nn_model = NearestNeighbors(
                n_neighbors=k, algorithm="auto", metric="euclidean"
            ).fit(positions_downsampled)
            _, indices = nn_model.kneighbors(positions_np)
            clusterer.labels_[non_clustered] = labels[indices[:, 0]]

        labels = clusterer.labels_
        print(f"done. Took {time.time()-start} seconds. Found {labels.max() + 1} clusters.")

        noise_mask = labels == -1
        if noise_mask.sum() != 0 and (labels>=0).sum() > 0:
            # if there is noise, but not all of it is noise, relabel the noise
            valid_mask = labels >=0
            valid_positions = positions[valid_mask]
            k = 1
            nn_model = NearestNeighbors(
                n_neighbors=k, algorithm="auto", metric="euclidean"
            ).fit(valid_positions)
            noise_positions = positions[noise_mask]
            _, indices = nn_model.kneighbors(noise_positions)
            # for now just pick the closest cluster
            noise_relabels = labels[valid_mask][indices[:, 0]]
            labels[noise_mask] = noise_relabels
            clusterer.labels_ = labels

        labels = clusterer.labels_

        colormap = self.colormap

        opacities = self.model.gauss_params['opacities'].detach()
        opacities[labels < 0] = -100  # hide unclustered gaussians
        self.model.gauss_params['opacities'] = torch.nn.Parameter(opacities.float())

        self.cluster_labels = torch.Tensor(labels)
        
        self.num_clusters.set_hidden(False)
        self.num_clusters.gui_handle.value = len(self.cluster_labels.unique())
        
        features_dc = self.model.gauss_params['features_dc'].detach()
        features_rest = self.model.gauss_params['features_rest'].detach()
        for c_id in range(0, labels.max() + 1):
            # set the colors of the gaussians accordingly using colormap from matplotlib
            cluster_mask = np.where(labels == c_id)
            features_dc[cluster_mask] = RGB2SH(colormap[c_id, :3].to(self.model.gauss_params['features_dc']))
            features_rest[cluster_mask] = 0

        self.model.gauss_params['features_dc'] = torch.nn.Parameter(self.model.gauss_params['features_dc'])
        self.model.gauss_params['features_rest'] = torch.nn.Parameter(self.model.gauss_params['features_rest'])

        self.cluster_scene_z.set_disabled(False)
        self.viewer_control.viewer._trigger_rerender()  # trigger viewer rerender
        self.click_gaussian.set_disabled(False)
        
        self.save_state(None)