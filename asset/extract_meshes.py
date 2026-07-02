"""Extract individual OBJ meshes from FMB STEP files.

Usage:
    uv run python asset/extract_meshes.py
"""

import os
import numpy as np
import cadquery as cq
import trimesh

ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
STEP_DIR = os.path.join(ASSET_DIR, "step")
OBJ_DIR = os.path.join(ASSET_DIR, "obj")

# All 54 peg solid indices inside peg.step.
# Mapping verified by aspect-ratio matching against the known medium-short indices
# from fmb-isaaclab (corrected from original FMB).
#
# peg.step solids 0-26: short (100mm), 27-53: long (150mm)
# Each shape appears in 3 sizes (S/M/L) × 2 lengths = 6 variants.
SHAPE_INDEX = {
    #                    short               long
    # shape:         S    M    L          S    M    L
    "circle":       (25,  10,  21,        49,  50,  37),
    "oval":         (24,   3,  20,        39,  46,  35),
    "rectangle":    (23,   9,   7,        32,  48,  52),
    "hexagon":      (18,  11,   6,        47,  53,  51),
    "arch":         ( 5,  19,  22,        45,  36,  38),
    "star":         ( 4,  17,   8,        33,  40,  34),
    "3prong":       (26,  13,   1,        31,  29,  27),
    "doublesquare": (12,  14,   0,        44,  42,  28),
    "squarecircle": (16,   2,  15,        30,  41,  43),
}

SIZES = ["S", "M", "L"]
LENGTHS = ["short", "long"]

MM = 0.001  # mm -> metres


def solid_to_trimesh(solid, tolerance=0.3):
    """Tessellate a CadQuery solid and return a trimesh.Trimesh (in metres, centered)."""
    bb = solid.BoundingBox()
    cx = (bb.xmin + bb.xmax) / 2.0
    cy = (bb.ymin + bb.ymax) / 2.0
    cz = bb.zmin  # Z-min = 0

    verts_cq, faces_cq = solid.tessellate(tolerance)
    verts = np.array([((p.x - cx) * MM, (p.y - cy) * MM, (p.z - cz) * MM) for p in verts_cq])
    faces = np.array(faces_cq)
    return trimesh.Trimesh(vertices=verts, faces=faces)


def main():
    os.makedirs(OBJ_DIR, exist_ok=True)

    # --- All 54 Pegs ---
    print("Loading peg.step ...")
    peg_cad = cq.importers.importStep(os.path.join(STEP_DIR, "peg.step"))
    peg_solids = peg_cad.solids().vals()
    print(f"  {len(peg_solids)} solids found\n")

    for shape, indices in SHAPE_INDEX.items():
        for li, length in enumerate(LENGTHS):
            for si, size in enumerate(SIZES):
                idx = indices[li * 3 + si]
                solid = peg_solids[idx]
                mesh = solid_to_trimesh(solid)
                name = f"peg_{shape}_{size}_{length}"
                out = os.path.join(OBJ_DIR, f"{name}.obj")
                mesh.export(out)
                bb = mesh.bounds
                sz = (bb[1] - bb[0]) * 1000
                print(f"  {name:35s}  solid[{idx:2d}]  {sz[0]:5.1f}x{sz[1]:5.1f}x{sz[2]:5.1f} mm")

    # --- Boards (single-object hole boards) ---
    print("\nLoading peg_board.step ...")
    board_cad = cq.importers.importStep(os.path.join(STEP_DIR, "peg_board.step"))
    board_solids = board_cad.solids().vals()
    print(f"  {len(board_solids)} solids found")

    for i, solid in enumerate(board_solids):
        mesh = solid_to_trimesh(solid)
        out = os.path.join(OBJ_DIR, f"hole_board_{i + 1}.obj")
        mesh.export(out)
        bb = mesh.bounds
        sz = (bb[1] - bb[0]) * 1000
        print(f"  hole_board_{i + 1}  {sz[0]:.0f}x{sz[1]:.0f}x{sz[2]:.0f} mm")

    # --- Multi-object assemblies ---
    for board_name in ["Board_1", "Board_2", "Board_3"]:
        print(f"\nLoading {board_name}.step ...")
        cad = cq.importers.importStep(os.path.join(STEP_DIR, f"{board_name}.step"))
        solids = cad.solids().vals()
        folder = os.path.join(OBJ_DIR, board_name.lower())
        os.makedirs(folder, exist_ok=True)
        for j, solid in enumerate(solids):
            mesh = solid_to_trimesh(solid)
            label = "base" if j == 0 else f"part{j}"
            out = os.path.join(folder, f"{board_name.lower()}_{label}.obj")
            mesh.export(out)
            bb = mesh.bounds
            sz = (bb[1] - bb[0]) * 1000
            print(f"  {label:6s}  {sz[0]:.0f}x{sz[1]:.0f}x{sz[2]:.0f} mm")

    print(f"\nDone. {len(SHAPE_INDEX) * 6} pegs + boards + assemblies in {OBJ_DIR}")


if __name__ == "__main__":
    main()
