"""Open FMB scene in the interactive MuJoCo viewer.

Usage:
    uv run python notebook/view_scene.py                         # default: M short
    uv run python notebook/view_scene.py scene_single_L_long     # specific scene
    uv run python notebook/view_scene.py scene_multi_1           # multi-object
"""

import sys
import mujoco
import mujoco.viewer

scene = sys.argv[1] if len(sys.argv) > 1 else "scene_single_M_short"
xml_path = f"asset/{scene}.xml"
print(f"Loading {xml_path}")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
if model.nkey:  # open in the baked "aligned" state
    mujoco.mj_resetDataKeyframe(model, data, 0)

mujoco.viewer.launch(model, data)
