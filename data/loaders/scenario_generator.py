"""
Realistic procedural LiDAR sequence generator with dynamic vehicles, pedestrians,
curbs, potholes, buildings, trees, and realistic multi-beam LiDAR beam geometry.
"""
from typing import List, Tuple
import numpy as np
from data.loaders.base_loader import BasePointCloudLoader, PointCloudFrame


class ProceduralScenarioGenerator(BasePointCloudLoader):
    """
    Generates realistic dynamic urban driving sequences with ground truth semantics,
    instance IDs, and realistic LiDAR scan patterns (64-beam Velodyne geometry).
    """

    def __init__(self, num_frames: int = 50, dt: float = 0.1):
        self.num_frames_val = num_frames
        self.dt = dt
        self.rng = np.random.RandomState(42)

        # Dynamic actors state
        # Actor: [type, [x0, y0, z0], [vx, vy, vz], [size_x, size_y, size_z], class_id]
        self.actors = [
            # Leading car moving forward at 12 m/s
            {"type": "car", "pos": np.array([20.0, -1.8, -0.2]), "vel": np.array([12.0, 0.0, 0.0]), "size": [4.2, 1.8, 1.4], "class_id": 4, "inst_id": 101},
            # Oncoming car moving towards ego in opposite lane at -14 m/s
            {"type": "car", "pos": np.array([65.0, 1.8, -0.2]), "vel": np.array([-14.0, 0.0, 0.0]), "size": [4.5, 1.9, 1.5], "class_id": 4, "inst_id": 102},
            # Pedestrian crossing road at x=12m
            {"type": "pedestrian", "pos": np.array([12.0, -4.5, 0.1]), "vel": np.array([0.0, 1.2, 0.0]), "size": [0.5, 0.5, 1.7], "class_id": 5, "inst_id": 103},
            # Cyclist riding along sidewalk at x=28m
            {"type": "cyclist", "pos": np.array([28.0, 4.2, 0.0]), "vel": np.array([4.0, 0.0, 0.0]), "size": [1.6, 0.6, 1.6], "class_id": 6, "inst_id": 104},
        ]

        # Static elements (Poles, Buildings, Trees, Walls)
        self.static_objects = self._generate_static_environment()

    def __len__(self) -> int:
        return self.num_frames_val

    def reset(self) -> None:
        self.rng = np.random.RandomState(42)

    def _generate_static_environment(self) -> List[dict]:
        objs = []
        # Lamp posts along both sidewalks every 15m
        for x in range(-20, 110, 15):
            objs.append({"type": "pole", "pos": [float(x), -4.8, 0.1], "radius": 0.15, "height": 5.0, "class_id": 9, "inst_id": 200 + x})
            objs.append({"type": "pole", "pos": [float(x), 4.8, 0.1], "radius": 0.15, "height": 5.0, "class_id": 9, "inst_id": 300 + x})

        # Trees
        for x in range(-15, 105, 20):
            objs.append({"type": "tree", "pos": [float(x + 5), -7.5, 0.1], "trunk_r": 0.3, "foliage_r": 2.2, "height": 6.5, "class_id": 8, "inst_id": 400 + x})
            objs.append({"type": "tree", "pos": [float(x + 8), 7.5, 0.1], "trunk_r": 0.3, "foliage_r": 2.2, "height": 6.5, "class_id": 8, "inst_id": 500 + x})

        # Buildings (facades)
        for x in range(-20, 100, 30):
            objs.append({"type": "building", "box": [x, -14.0, 0.0, 26.0, 8.0, 10.0], "class_id": 7, "inst_id": 600 + x})
            objs.append({"type": "building", "box": [x, 14.0, 0.0, 26.0, 8.0, 12.0], "class_id": 7, "inst_id": 700 + x})

        return objs

    def get_frame(self, index: int) -> PointCloudFrame:
        time_elapsed = index * self.dt
        ego_speed = 8.0  # Ego moves forward at 8 m/s (approx 29 km/h)
        ego_x = ego_speed * time_elapsed

        points_list = []
        semantics_list = []
        instances_list = []
        intensity_list = []

        # 1. Generate Road, Curbs, Sidewalks, and Terrain relative to Ego Vehicle
        # Grid sample points representing sensor field
        road_pts, road_sem, road_inst, road_int = self._generate_ground_surface(ego_x)
        points_list.append(road_pts)
        semantics_list.append(road_sem)
        instances_list.append(road_inst)
        intensity_list.append(road_int)

        # 2. Add Static Environment Points (relative to ego)
        static_pts, static_sem, static_inst, static_int = self._generate_static_points(ego_x)
        if len(static_pts) > 0:
            points_list.append(static_pts)
            semantics_list.append(static_sem)
            instances_list.append(static_inst)
            intensity_list.append(static_int)

        # 3. Add Dynamic Actors Points
        for actor in self.actors:
            curr_world_pos = actor["pos"] + actor["vel"] * time_elapsed
            # Transform to ego sensor frame
            curr_sensor_pos = curr_world_pos.copy()
            curr_sensor_pos[0] -= ego_x

            # Only render if within sensor range (-30m to +100m)
            if -30.0 <= curr_sensor_pos[0] <= 100.0 and abs(curr_sensor_pos[1]) <= 50.0:
                act_pts, act_sem, act_inst, act_int = self._sample_box_actor(curr_sensor_pos, actor["size"], actor["class_id"], actor["inst_id"])
                if len(act_pts) > 0:
                    points_list.append(act_pts)
                    semantics_list.append(act_sem)
                    instances_list.append(act_inst)
                    intensity_list.append(act_int)

        # Concatenate all
        all_pts = np.vstack(points_list).astype(np.float32)
        all_sem = np.concatenate(semantics_list).astype(np.uint32)
        all_inst = np.concatenate(instances_list).astype(np.uint32)
        all_int = np.concatenate(intensity_list).astype(np.float32)

        # Simulate LiDAR range and ray-density drop-off with distance
        dist = np.linalg.norm(all_pts[:, :2], axis=1)
        valid_mask = (dist >= 0.5) & (dist <= 100.0)
        
        # Realistic distance-dependent sampling: point density naturally decreases as 1/r
        keep_prob = np.clip(18.0 / (dist + 1e-3), 0.05, 1.0)
        rand_vals = self.rng.uniform(0.0, 1.0, size=len(all_pts))
        valid_mask = valid_mask & (rand_vals <= keep_prob)

        all_pts = all_pts[valid_mask]
        all_sem = all_sem[valid_mask]
        all_inst = all_inst[valid_mask]
        all_int = all_int[valid_mask]

        # Add slight LiDAR sensor noise (Gaussian sigma=0.015m)
        all_pts += self.rng.normal(0.0, 0.015, size=all_pts.shape).astype(np.float32)

        # Ego Pose
        ego_pose = np.eye(4, dtype=np.float32)
        ego_pose[0, 3] = ego_x

        return PointCloudFrame(
            frame_id=index,
            timestamp=time_elapsed,
            points=all_pts,
            intensities=all_int,
            semantic_labels=all_sem,
            instance_labels=all_inst,
            ego_pose=ego_pose,
            metadata={"ego_speed_mps": ego_speed, "ego_x": ego_x}
        )

    def _generate_ground_surface(self, ego_x: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Generate polar/ring LiDAR scan on the ground plane z = -1.73m (sensor mounted at 1.73m height)
        sensor_z = 0.0
        ground_z_base = -1.73

        # Sample rings
        num_rings = 48
        elevations = np.linspace(-24.0, 2.0, num_rings) * np.pi / 180.0
        azimuths = np.linspace(-np.pi, np.pi, 360, endpoint=False)

        pts = []
        sems = []
        insts = []
        intens = []

        for elev in elevations:
            if np.sin(elev) >= -0.01:
                continue  # Upward pointing rays hit sky unless intersecting objects
            # Ground intersection distance
            r = -ground_z_base / (-np.sin(elev))
            if r <= 0 or r > 100.0:
                continue

            x = r * np.cos(elev) * np.cos(azimuths)
            y = r * np.cos(elev) * np.sin(azimuths)
            z = np.full_like(x, ground_z_base)

            # Assign terrain classes based on lateral offset y and surface features
            sem = np.zeros(len(x), dtype=np.uint32)
            inst = np.zeros(len(x), dtype=np.uint32)
            inten = np.full(len(x), 0.35, dtype=np.float32)

            for i in range(len(x)):
                yi = y[i]
                xi = x[i]
                abs_y = abs(yi)
                world_x = xi + ego_x

                if abs_y <= 3.6:
                    # Drivable road lane
                    sem[i] = 1  # drivable_road
                    inten[i] = 0.4
                    # Add lane marking high reflectivity
                    if abs(abs_y - 1.8) < 0.12 or abs_y < 0.08:
                        inten[i] = 0.85
                    # Add synthetic pothole at x=8m, y=-1.5m in Near Field
                    if 6.5 <= xi <= 8.5 and -2.2 <= yi <= -0.8:
                        sem[i] = 12  # pothole_irregular
                        z[i] -= 0.09  # Depression
                elif abs_y <= 3.85:
                    # Curb edge (step height +0.16m)
                    sem[i] = 11  # curb
                    z[i] += 0.16
                    inten[i] = 0.5
                elif abs_y <= 6.0:
                    # Sidewalk
                    sem[i] = 2   # sidewalk
                    z[i] += 0.16
                    inten[i] = 0.45
                else:
                    # Rough terrain / grass
                    sem[i] = 3   # rough_terrain
                    z[i] += 0.14 + 0.04 * np.sin(xi * 0.5)
                    inten[i] = 0.2

            pts.append(np.column_stack([x, y, z]))
            sems.append(sem)
            insts.append(inst)
            intens.append(inten)

        if pts:
            return np.vstack(pts), np.concatenate(sems), np.concatenate(insts), np.concatenate(intens)
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0)

    def _generate_static_points(self, ego_x: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pts = []
        sems = []
        insts = []
        intens = []

        ground_z_base = -1.73

        for obj in self.static_objects:
            obj_type = obj["type"]
            c_id = obj["class_id"]
            i_id = obj["inst_id"]

            if obj_type == "pole":
                px, py, pz = obj["pos"]
                rel_x = px - ego_x
                if -25.0 <= rel_x <= 95.0:
                    # Sample cylinder for pole
                    z_vals = np.linspace(ground_z_base + 0.16, ground_z_base + obj["height"], 25)
                    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
                    for zv in z_vals:
                        cx = rel_x + obj["radius"] * np.cos(angles)
                        cy = py + obj["radius"] * np.sin(angles)
                        cz = np.full_like(cx, zv)
                        pts.append(np.column_stack([cx, cy, cz]))
                        sems.append(np.full(len(cx), c_id, dtype=np.uint32))
                        insts.append(np.full(len(cx), i_id, dtype=np.uint32))
                        intens.append(np.full(len(cx), 0.7, dtype=np.float32))

            elif obj_type == "tree":
                tx, ty, tz = obj["pos"]
                rel_x = tx - ego_x
                if -25.0 <= rel_x <= 95.0:
                    # Foliage sphere
                    n_foliage = 45
                    u = self.rng.uniform(0, 1, n_foliage)
                    v = self.rng.uniform(0, 1, n_foliage)
                    theta = u * 2.0 * np.pi
                    phi = np.arccos(2.0 * v - 1.0)
                    r = obj["foliage_r"] * (self.rng.uniform(0.7, 1.0, n_foliage) ** (1/3))
                    fx = rel_x + r * np.sin(phi) * np.cos(theta)
                    fy = ty + r * np.sin(phi) * np.sin(theta)
                    fz = ground_z_base + 3.0 + r * np.cos(phi)
                    pts.append(np.column_stack([fx, fy, fz]))
                    sems.append(np.full(len(fx), c_id, dtype=np.uint32))
                    insts.append(np.full(len(fx), i_id, dtype=np.uint32))
                    intens.append(np.full(len(fx), 0.3, dtype=np.float32))

            elif obj_type == "building":
                bx, by, bz, bw, bl, bh = obj["box"]
                rel_x = bx - ego_x
                if -35.0 <= rel_x <= 95.0:
                    # Facade surface visible to sensor
                    y_face = by - np.sign(by) * (bl / 2.0)
                    xs = np.linspace(rel_x - bw/2.0, rel_x + bw/2.0, 20)
                    zs = np.linspace(ground_z_base, ground_z_base + bh, 15)
                    X_grid, Z_grid = np.meshgrid(xs, zs)
                    Y_grid = np.full_like(X_grid, y_face)
                    flat_x = X_grid.flatten()
                    flat_y = Y_grid.flatten()
                    flat_z = Z_grid.flatten()
                    pts.append(np.column_stack([flat_x, flat_y, flat_z]))
                    sems.append(np.full(len(flat_x), c_id, dtype=np.uint32))
                    insts.append(np.full(len(flat_x), i_id, dtype=np.uint32))
                    intens.append(np.full(len(flat_x), 0.5, dtype=np.float32))

        if pts:
            return np.vstack(pts), np.concatenate(sems), np.concatenate(insts), np.concatenate(intens)
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0)

    def _sample_box_actor(self, center: np.ndarray, size: List[float], class_id: int, inst_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sx, sy, sz = size
        ground_z_base = -1.73
        base_z = ground_z_base + 0.1 + sz / 2.0
        cx, cy = center[0], center[1]

        # Sample box surface facing vehicle
        n_pts = 60 if class_id == 4 else 25
        # Front/rear/side faces
        x_pts = self.rng.uniform(cx - sx/2.0, cx + sx/2.0, n_pts)
        y_pts = self.rng.uniform(cy - sy/2.0, cy + sy/2.0, n_pts)
        z_pts = self.rng.uniform(base_z - sz/2.0, base_z + sz/2.0, n_pts)

        pts = np.column_stack([x_pts, y_pts, z_pts])
        sem = np.full(len(pts), class_id, dtype=np.uint32)
        inst = np.full(len(pts), inst_id, dtype=np.uint32)
        inten = np.full(len(pts), 0.75 if class_id == 4 else 0.45, dtype=np.float32)
        return pts, sem, inst, inten
