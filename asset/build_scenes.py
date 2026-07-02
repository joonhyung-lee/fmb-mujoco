"""Generate FMB MuJoCo scenes with baked "aligned" keyframes.

Single scenes: 9 pegs seated in their matching holes (hybrid board collision).
Multi scenes: 4 parts interlocked on the base at their CAD assembled poses.

Usage:
    uv run python asset/build_scenes.py            # generate + verify
    uv run python asset/build_scenes.py --render   # also write media/scenes/*.png
"""

import os
import sys
import glob
import numpy as np
import trimesh
import mujoco

ASSET = os.path.dirname(os.path.abspath(__file__))
OBJ = os.path.join(ASSET, "obj")

SHAPES = ["circle", "oval", "rectangle", "hexagon", "arch",
          "star", "3prong", "doublesquare", "squarecircle"]

# Approximate cell anchors of the 3x3 hole grid (same on all boards). Used
# only for grouping measured hole regions and placing divider walls — exact
# per-board hole centers are measured from the mesh (measure_hole_centers).
CELL_XY = {
    "circle": (-0.009, 0.004), "oval": (-0.006, 0.070), "rectangle": (0.066, 0.070),
    "hexagon": (-0.073, -0.068), "arch": (0.067, -0.077), "star": (-0.074, 0.001),
    "3prong": (-0.007, -0.070), "doublesquare": (0.065, 0.003), "squarecircle": (-0.073, 0.071),
}

BOARD_FOR_SIZE = {"S": "hole_board_3", "M": "hole_board_2", "L": "hole_board_1"}

# Pegs extracted from the STEP with a solid base at the bottom and insertion
# posts pointing up; they insert upside-down. 180deg flip about X matches the
# board hole patterns (3prong: two-down-one-up; squarecircle: square left).
FLIPPED = {"3prong", "doublesquare", "squarecircle"}
FLIP_QUAT = "0 1 0 0"

PEG_RGBA = {
    "circle": (0.55, 0.27, 0.07), "oval": (0.18, 0.35, 0.65), "rectangle": (0.18, 0.30, 0.60),
    "hexagon": (0.10, 0.55, 0.15), "arch": (1.00, 0.85, 0.00), "star": (0.05, 0.12, 0.50),
    "3prong": (0.65, 0.15, 0.15), "doublesquare": (0.50, 0.10, 0.65), "squarecircle": (0.85, 0.10, 0.10),
}

# Assembled part offsets relative to base center (metres), measured from
# Board_{1,2,3}.step absolute solid placements (z = part zmin in STEP frame).
ASSEMBLY_OFFSETS = {
    1: [(-0.068, 0.0, 0.005), (0.0, 0.0, 0.040), (0.0, 0.0, 0.005), (0.068, 0.0, 0.005)],
    2: [(0.0, 0.0, 0.030), (0.0, 0.0, 0.005), (-0.075, 0.0, 0.015), (0.075, 0.0, 0.015)],
    3: [(-0.0365, 0.0, 0.040), (0.0365, 0.0, 0.040), (0.0, -0.0515, 0.025), (0.0, 0.0515, 0.025)],
}
PART_RGBA = [(0.85, 0.30, 0.20), (0.20, 0.60, 0.30), (0.90, 0.75, 0.10), (0.60, 0.25, 0.70)]

BOARD_Z = 0.8
SPAWN_Z = BOARD_Z + 0.011  # peg bottoms 1mm above the hole floor (board is 50mm
                           # tall, holes span z=10..50mm). Spawn seated: a long
                           # slide-in jams tight holes (mu=1 self-locking against
                           # ~1mm CoACD wall bulge); a 1mm drop cannot.

# Primitive containment walls: bottom plate + outer walls (fixed) and 3x3 grid
# dividers generated per board, trimmed where they'd intersect a measured hole
# (large L holes reach into the nominal divider lines). 3mm half-width, invisible.
DIVIDERS_V = (-0.040, 0.0295)  # vertical divider x positions
DIVIDERS_H = (-0.035, 0.036)   # horizontal divider y positions


def _subtract(spans, lo, hi):
    out = []
    for a, b in spans:
        if hi <= a or lo >= b:
            out.append((a, b))
        else:
            if lo > a:
                out.append((a, lo))
            if hi < b:
                out.append((hi, b))
    return out


def board_walls(board):
    _, boxes, _, _, _ = measure_holes(board)
    g = ['      <geom type="box" pos="0 0 0.005" size="0.115 0.115 0.005" rgba="0 0 0 0"/>',
         '      <geom type="box" pos="-0.112 0 0.025" size="0.003 0.115 0.025" rgba="0 0 0 0"/>',
         '      <geom type="box" pos="0.112 0 0.025" size="0.003 0.115 0.025" rgba="0 0 0 0"/>',
         '      <geom type="box" pos="0 0.112 0.025" size="0.115 0.003 0.025" rgba="0 0 0 0"/>',
         '      <geom type="box" pos="0 -0.112 0.025" size="0.115 0.003 0.025" rgba="0 0 0 0"/>']
    pad = 0.002
    for xv in DIVIDERS_V:
        spans = [(-0.115, 0.115)]
        for b in boxes.values():
            if b[0] - pad < xv + 0.003 and b[1] + pad > xv - 0.003:
                spans = _subtract(spans, b[2] - pad, b[3] + pad)
        for y0, y1 in spans:
            if y1 - y0 > 0.006:
                g.append(f'      <geom type="box" pos="{xv} {(y0 + y1) / 2:.4f} 0.025" '
                         f'size="0.003 {(y1 - y0) / 2:.4f} 0.025" rgba="0 0 0 0"/>')
    for yh in DIVIDERS_H:
        spans = [(-0.115, 0.115)]
        for b in boxes.values():
            if b[2] - pad < yh + 0.003 and b[3] + pad > yh - 0.003:
                spans = _subtract(spans, b[0] - pad, b[1] + pad)
        for x0, x1 in spans:
            if x1 - x0 > 0.006:
                g.append(f'      <geom type="box" pos="{(x0 + x1) / 2:.4f} {yh} 0.025" '
                         f'size="{(x1 - x0) / 2:.4f} 0.003 0.025" rgba="0 0 0 0"/>')
    return "\n".join(g) + "\n"

HEADER = """<mujoco model="{model}">
  <compiler angle="radian" autolimits="true" meshdir="obj" texturedir="obj"/>
  <option integrator="RK4"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <include file="floor_isaac.xml"/>
  <asset>
    <texture type="2d" name="light_wood" file="light_wood_v3.png"/>
    <material name="wt" texture="light_wood" specular="0.5" shininess="0.5" rgba="0.5 0.5 0.5 1"/>
{meshes}  </asset>
  <worldbody>
    <body name="table"><geom type="box" size="0.4 0.4 0.4" pos="0 0 0.4" material="wt" mass="100"/></body>
{bodies}  </worldbody>
</mujoco>
"""


def peg_dims(shape, size):
    """Peg XY footprint (metres) from the short-variant visual mesh."""
    m = trimesh.load(os.path.join(OBJ, f"peg_{shape}_{size}_short", f"peg_{shape}_{size}_short.obj"))
    ext = m.bounds[1] - m.bounds[0]
    return ext[0], ext[1]


_HOLE_CACHE = {}


def measure_holes(board):
    """Exact per-shape hole bbox centers + bboxes, measured from the board mesh.

    Cross-section at z=30mm (mid-hole) -> scanline parity fill -> connected
    empty regions -> grouped by nearest CELL_XY anchor (compound holes like
    3prong/squarecircle split into several regions) -> union bbox.
    The approximate CELL_XY table is up to ~11mm off the true centers, which
    varies per board because compound-hole bbox centers shift with hole size.
    Scan window ±114.5mm: board spans ±115mm and board_1's doublesquare hole
    reaches x=106.6mm — a smaller window border-excludes that region.
    """
    if board in _HOLE_CACHE:
        return _HOLE_CACHE[board]
    res = 0.0005
    m = trimesh.load(os.path.join(OBJ, board, f"{board}.obj"))
    segs = trimesh.intersections.mesh_plane(m, plane_normal=[0, 0, 1], plane_origin=[0, 0, 0.030])[:, :, :2]
    xs = np.arange(-0.1145, 0.1145, res)
    ys = np.arange(-0.1145, 0.1145, res)
    inside = np.zeros((len(xs), len(ys)), bool)
    p0, p1 = segs[:, 0], segs[:, 1]
    for j, y in enumerate(ys):
        c = (p0[:, 1] <= y) != (p1[:, 1] <= y)
        if not c.any():
            continue
        a, b = p0[c], p1[c]
        xc = np.sort(a[:, 0] + (y - a[:, 1]) / (b[:, 1] - a[:, 1]) * (b[:, 0] - a[:, 0]))
        for k in range(0, len(xc) - 1, 2):
            inside[:, j] |= (xs >= xc[k]) & (xs < xc[k + 1])
    # connected empty regions (holes); regions touching the border are outside
    from collections import deque
    empty = ~inside
    lab = np.zeros(empty.shape, int)
    cur = 0
    for i in range(empty.shape[0]):
        for j in range(empty.shape[1]):
            if empty[i, j] and lab[i, j] == 0:
                cur += 1
                q = deque([(i, j)])
                lab[i, j] = cur
                while q:
                    a, b = q.popleft()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < lab.shape[0] and 0 <= nb < lab.shape[1] \
                                and empty[na, nb] and lab[na, nb] == 0:
                            lab[na, nb] = cur
                            q.append((na, nb))
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    boxes = {}  # shape -> [xmin, xmax, ymin, ymax]
    for k in range(1, cur + 1):
        if k in border:
            continue
        ii, jj = np.where(lab == k)
        if len(ii) < 40:
            continue
        cx, cy = xs[ii].mean(), ys[jj].mean()
        shape = min(CELL_XY, key=lambda s: (CELL_XY[s][0] - cx) ** 2 + (CELL_XY[s][1] - cy) ** 2)
        bb = [xs[ii.min()], xs[ii.max()], ys[jj.min()], ys[jj.max()]]
        if shape in boxes:
            o = boxes[shape]
            boxes[shape] = [min(o[0], bb[0]), max(o[1], bb[1]), min(o[2], bb[2]), max(o[3], bb[3])]
        else:
            boxes[shape] = bb
    assert set(boxes) == set(SHAPES), f"{board}: found holes for {sorted(boxes)}"
    centers = {s: ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2) for s, b in boxes.items()}
    # "deep hole" mask: accepted hole regions eroded by 2mm — used to classify
    # collision hulls as plugs (vertices deep inside a hole) vs wall geometry
    hole_mask = np.isin(lab, [k for k in range(1, cur + 1)
                              if k not in border and (lab == k).sum() >= 40])
    for _ in range(4):  # 4 x 0.5mm = 2mm erosion
        hole_mask = (hole_mask
                     & np.roll(hole_mask, 1, 0) & np.roll(hole_mask, -1, 0)
                     & np.roll(hole_mask, 1, 1) & np.roll(hole_mask, -1, 1))
    _HOLE_CACHE[board] = (centers, boxes, hole_mask, xs, ys)
    return _HOLE_CACHE[board]


def filter_board_hulls(board):
    """Drop CoACD hulls that plug a hole cavity; keep all wall/rim/floor hulls.

    A hull is a plug iff it has a vertex deep inside a hole (2mm-eroded hole
    mask) above the hole floor (z > 12mm). Wall hulls only graze the boundary,
    so real hole walls survive and pegs get true lateral constraint.
    """
    _, _, hole_mask, xs, ys = measure_holes(board)
    kept = []
    for f in sorted(glob.glob(os.path.join(OBJ, board, f"{board}_collision_*.obj"))):
        v = trimesh.load(f).vertices
        v = v[v[:, 2] > 0.012]
        i = np.searchsorted(xs, v[:, 0])
        j = np.searchsorted(ys, v[:, 1])
        ok = (i > 0) & (i < len(xs)) & (j > 0) & (j < len(ys))
        if not hole_mask[i[ok] - 1, j[ok] - 1].any():
            kept.append(os.path.basename(f))
    return kept


def mesh_block(entries):
    return "".join(f'    <mesh name="{n}" file="{f}"/>\n' for n, f in entries)


def post_spawn(shape, size, length, hx, hy):
    """Spawn pos for a flipped peg: post-union center (top slice) over the
    hole center, tips at SPAWN_Z (1mm above the hole floor). Flip about X maps
    (x,y)->(x,-y) and puts the body origin at the peg top."""
    name = f"peg_{shape}_{size}_{length}"
    m = trimesh.load(os.path.join(OBJ, name, f"{name}.obj"))
    zmax = m.bounds[1][2]
    pts = trimesh.intersections.mesh_plane(m, [0, 0, 1], [0, 0, zmax - 0.005]).reshape(-1, 3)[:, :2]
    cx, cy = (pts.min(0) + pts.max(0)) / 2
    return hx - cx, hy + cy, SPAWN_Z + zmax


def peg_body(shape, size, length, pos, quat=None):
    name = f"peg_{shape}_{size}_{length}"
    r, g, b = PEG_RGBA[shape]
    geoms = [f'      <geom mesh="{name}" group="2" type="mesh" contype="0" conaffinity="0" rgba="{r} {g} {b} 1"/>']
    for f in sorted(glob.glob(os.path.join(OBJ, name, f"{name}_collision_*.obj"))):
        cname = f"{name}_c{os.path.basename(f).split('_')[-1].split('.')[0]}"
        geoms.append(f'      <geom mesh="{cname}" type="mesh" rgba="0 0 0 0"/>')
    q = f' quat="{quat}"' if quat else ""
    body = (f'    <body name="{shape}" pos="{pos[0]} {pos[1]} {pos[2]}"{q}>\n'
            '      <joint type="free"/>\n' + "\n".join(geoms) + "\n    </body>\n")
    meshes = [(name, f"{name}/{name}.obj")]
    for f in sorted(glob.glob(os.path.join(OBJ, name, f"{name}_collision_*.obj"))):
        idx = os.path.basename(f).split("_")[-1].split(".")[0]
        meshes.append((f"{name}_c{idx}", f"{name}/{os.path.basename(f)}"))
    return body, meshes


def build_single(size, length):
    board = BOARD_FOR_SIZE[size]
    hulls = filter_board_hulls(board)
    meshes = [("board_vis", f"{board}/{board}.obj")]
    meshes += [(f"bc{i}", f"{board}/{h}") for i, h in enumerate(hulls)]

    bodies = f'    <body name="board" pos="0 0 {BOARD_Z}">\n'
    bodies += '      <geom mesh="board_vis" group="2" type="mesh" contype="0" conaffinity="0" rgba="0.2 0.25 0.55 1"/>\n'
    bodies += "".join(f'      <geom mesh="bc{i}" type="mesh" rgba="0 0 0 0"/>\n' for i in range(len(hulls)))
    bodies += board_walls(board) + "    </body>\n"

    holes, _, _, _, _ = measure_holes(board)
    for shape in SHAPES:
        hx, hy = holes[shape]
        if shape in FLIPPED:
            b, m = peg_body(shape, size, length, post_spawn(shape, size, length, hx, hy), FLIP_QUAT)
        else:
            b, m = peg_body(shape, size, length, (hx, hy, SPAWN_Z))
        bodies += b
        meshes += m

    xml = HEADER.format(model=f"FMB {size} {length}", meshes=mesh_block(meshes), bodies=bodies)
    path = os.path.join(ASSET, f"scene_single_{size}_{length}.xml")
    with open(path, "w") as fp:
        fp.write(xml)
    return path, len(hulls)


def build_multi(n):
    base = f"board_{n}_base"
    meshes = [(base, f"board_{n}/{base}/{base}.obj")]
    base_hulls = sorted(glob.glob(os.path.join(OBJ, f"board_{n}", base, f"{base}_collision_*.obj")))
    meshes += [(f"base_c{i}", f"board_{n}/{base}/{os.path.basename(f)}") for i, f in enumerate(base_hulls)]

    bodies = f'    <body name="base" pos="0 0 {BOARD_Z}">\n'
    bodies += f'      <geom mesh="{base}" group="2" type="mesh" contype="0" conaffinity="0" rgba="0.2 0.25 0.55 1"/>\n'
    bodies += "".join(f'      <geom mesh="base_c{i}" type="mesh" rgba="0 0 0 0"/>\n' for i in range(len(base_hulls)))
    bodies += "    </body>\n"

    for j, (dx, dy, dz) in enumerate(ASSEMBLY_OFFSETS[n], start=1):
        part = f"board_{n}_part{j}"
        r, g, b = PART_RGBA[j - 1]
        meshes.append((part, f"board_{n}/{part}/{part}.obj"))
        geoms = [f'      <geom mesh="{part}" group="2" type="mesh" contype="0" conaffinity="0" rgba="{r} {g} {b} 1"/>']
        for f in sorted(glob.glob(os.path.join(OBJ, f"board_{n}", part, f"{part}_collision_*.obj"))):
            idx = os.path.basename(f).split("_")[-1].split(".")[0]
            meshes.append((f"{part}_c{idx}", f"board_{n}/{part}/{os.path.basename(f)}"))
            geoms.append(f'      <geom mesh="{part}_c{idx}" type="mesh" rgba="0 0 0 0"/>')
        # ponytail: damping kills frictionless spin on point contacts (condim=3 has no torsional friction)
        bodies += (f'    <body name="part{j}" pos="{dx} {dy} {BOARD_Z + dz + 0.0005}">\n'
                   '      <joint type="free" damping="0.005"/>\n' + "\n".join(geoms) + "\n    </body>\n")

    xml = HEADER.format(model=f"FMB Multi {n}", meshes=mesh_block(meshes), bodies=bodies)
    path = os.path.join(ASSET, f"scene_multi_{n}.xml")
    with open(path, "w") as fp:
        fp.write(xml)
    return path


def settle_and_bake(path, nsteps=5000, two_phase=False):
    """Simulate to rest, bake qpos as an 'aligned' keyframe into the XML.

    two_phase: heavy-damping descent first, then relax at real damping —
    keeps assembly parts from spinning/wedging while they sink ~0.5mm into
    their pockets. NOT for pegs: they must fall freely ~45mm to insert, and
    damped descent leaves them hung on hole rims.
    """
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    if two_phase:
        damp0 = model.dof_damping.copy()
        model.dof_damping[:] = 1.0
        for _ in range(nsteps // 2):
            mujoco.mj_step(model, data)
        model.dof_damping[:] = damp0
        data.qvel[:] = 0
    for _ in range(nsteps):
        mujoco.mj_step(model, data)
    qpos = " ".join(f"{q:.6g}" for q in data.qpos)
    with open(path) as fp:
        xml = fp.read()
    xml = xml.replace("</mujoco>",
                      f'  <keyframe><key name="aligned" qpos="{qpos}"/></keyframe>\n</mujoco>')
    with open(path, "w") as fp:
        fp.write(xml)
    return data


def verify(path, nsteps=2000):
    """From the keyframe, ensure every free body stays put."""
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    q0 = data.qpos.copy()
    for _ in range(nsteps):
        mujoco.mj_step(model, data)
    ok = True
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        adr = model.jnt_qposadr[j]
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[j])
        dz = abs(data.qpos[adr + 2] - q0[adr + 2])
        dq = np.abs(data.qpos[adr + 3:adr + 7] - q0[adr + 3:adr + 7]).max()
        z = data.qpos[adr + 2]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, data.qpos[adr + 3:adr + 7])
        up = -1.0 if body in FLIPPED else 1.0  # flipped pegs hang upside-down
        tilt = np.degrees(np.arccos(np.clip(up * R.reshape(3, 3)[2, 2], -1, 1)))
        # ponytail: dq 0.003 ~ 0.35deg — loose-fit pegs rattle that much in their holes
        flag = "" if (dz < 0.001 and dq < 0.003 and tilt < 5) else "  <-- DRIFT"
        if flag:
            ok = False
        print(f"    {body:14s} z={z:.4f}  dz={dz * 1000:.2f}mm  dq={dq:.5f}{flag}")
    return ok


def render(path, out):
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.85]
    cam.distance = 0.55
    cam.azimuth = 135
    cam.elevation = -60
    r = mujoco.Renderer(model, width=960, height=720)
    r.update_scene(data, camera=cam)
    from PIL import Image
    Image.fromarray(r.render()).save(out)
    r.close()


def main():
    do_render = "--render" in sys.argv
    media = os.path.join(os.path.dirname(ASSET), "media", "scenes")

    paths = []
    for size in ["S", "M", "L"]:
        for length in ["short", "long"]:
            path, nh = build_single(size, length)
            print(f"{os.path.basename(path)}: {nh} rim hulls")
            paths.append(path)
    for n in [1, 2, 3]:
        paths.append(build_multi(n))
        print(f"scene_multi_{n}.xml")

    print("\nSettling + baking keyframes ...")
    all_ok = True
    for path in paths:
        settle_and_bake(path, two_phase="multi" in os.path.basename(path))
        print(f"  {os.path.basename(path)}")
        all_ok &= verify(path)

    if do_render:
        os.makedirs(media, exist_ok=True)
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            render(path, os.path.join(media, f"{name}.png"))
            print(f"  rendered {name}.png")

    print("\nALL STABLE" if all_ok else "\nDRIFT DETECTED — inspect above")


if __name__ == "__main__":
    main()
