import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

YELLOW = (1.0, 1.0, 0.0)
RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


FORCE_MARKER_Z_CFG = VisualizationMarkersCfg(
    markers={
        "arrow_1": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.1, 0.1, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=YELLOW),
        ),
    }
)

ACC_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow_1": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.1, 0.1, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GREEN),
        ),
    }
)

DRONE_POS_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "current_pos_marker": sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GREEN),
        ),
        "goal_pos_marker_1": sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=RED),
        ),
        "goal_pos_marker_2": sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=YELLOW),
        ),
        "goal_pos_marker_3": sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BLUE),
        ),
    }
)

ORIENTATION_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame_1": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
        ),
    }
)
