"""Clean up the Meshy rocket and fit it to the simulated vehicle (data/vehicle.json).

    blender -b --factory-startup -P tools/rocket_model/process_meshy.py -- <in.glb> <out.glb>

Input: Meshy image-to-3D export (Z-up in Blender, ~1.9 units tall, nozzle exits at the bottom).
Output: four objects in body coordinates (metres, +Y = nose after glTF export, y = 0 at the stage-1
nozzle exit plane):  S1 (engine section, tank, interstage), S1_Nozzles, S2, FairingL (x >= 0), FairingR.

Steps: weld, delete interior shells, decimate (UV seams protected), record the nine nozzle exits,
lengthen the stage-1 and stage-2 barrels by repeating plain bands (no texture stretch), piecewise-scale each section to the vehicle geometry, widen the fairing to its 2.6 m radius,
split at the stage boundaries and down the fairing seam, then export.
"""
import json
import math
import sys

import bmesh
import bpy
import numpy as np

IN, OUT = sys.argv[-2], sys.argv[-1]
TARGET_FACES = 90_000
PART_FACES = {"S1": 52_000, "S1_Nozzles": 30_000}   # phone budget

# Meshy-space landmarks (measured from orthographic renders of the export)
Z_EXIT = -0.952          # nozzle exit plane
Z_NOZ = -0.860           # top of the nozzle bells (split S1 / S1_Nozzles here)
OVERLAP = 0.003          # repeated bands overlap so their joints never open a gap
Z_TANK0 = -0.760         # bottom rivet belt of the stage-1 tank
Z_INTER0 = 0.035         # stage-1 tank top / interstage bottom
Z_INTER1 = 0.235         # interstage top / stage-2 bottom (just above the texture's ragged black edge)
Z_FAIR = 0.511           # stage-2 top / fairing base
Z_FLARE = 0.550          # top of the fairing boat-tail flare
X_SEAM = -0.005          # fairing split seam (plane x = const)
R_BODY = 0.0756          # fitted body radius
S1_BAND, S1_COPIES = (-0.50, -0.14), 2
S2_BAND, S2_COPIES = (0.26, 0.46), 1

# vehicle geometry (metres)
K = 1.85 / R_BODY                          # Meshy units -> metres
Y_TANK0, Y_INTER0, Y_INTER1, Y_FAIR, Y_NOSE = None, 41.8, 45.0, 58.0, 70.0
FAIR_R = 2.6
EXIT_R = math.sqrt(0.8364 / math.pi)   # stage-1 nozzle exit radius (vehicle.json ae_m2)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=IN)
obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
me = obj.data
print("input faces", len(me.polygons))

# ---------------------------------------------------------------- weld + interior shells
bm = bmesh.new()
bm.from_mesh(me)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
bm.faces.ensure_lookup_table()
cen = np.array([f.calc_center_median()[:] for f in bm.faces])
rc = np.hypot(cen[:, 0], cen[:, 1])
bins = np.floor((cen[:, 2] - Z_EXIT) / 0.005).astype(int)
r90 = {}
for b in np.unique(bins):
    r90[b] = np.percentile(rc[bins == b], 90)
inner = [f for i, f in enumerate(bm.faces) if cen[i, 2] > Z_TANK0 + 0.004 and rc[i] < 0.8 * r90[bins[i]]]
bmesh.ops.delete(bm, geom=inner, context="FACES")
loose = [v for v in bm.verts if not v.link_faces]
bmesh.ops.delete(bm, geom=loose, context="VERTS")
bm.to_mesh(me)
bm.free()
print("after interior removal", len(me.polygons))

# ---------------------------------------------------------------- decimate
def protect_uv_seams(o):
    """Vertex group of UV-island border verts; decimating across them smears the texture gutter."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    uv = bm.loops.layers.uv.active
    border = set()
    for e in bm.edges:
        if len(e.link_loops) != 2:
            border.update(v.index for v in e.verts)
            continue
        l1, l2 = e.link_loops
        a1, b1 = l1[uv].uv, l1.link_loop_next[uv].uv
        a2, b2 = l2.link_loop_next[uv].uv, l2[uv].uv     # opposite winding
        if (a1 - a2).length > 1e-5 or (b1 - b2).length > 1e-5:
            border.update(v.index for v in e.verts)
    bm.free()
    vg = o.vertex_groups.get("uv_seams") or o.vertex_groups.new(name="uv_seams")
    vg.add(list(border), 1.0, "REPLACE")
    return vg


def decimate(o, faces):
    if len(o.data.polygons) <= faces:
        return
    vg = protect_uv_seams(o)
    m = o.modifiers.new("dec", "DECIMATE")
    m.ratio = faces / len(o.data.polygons)
    m.use_collapse_triangulate = True
    m.vertex_group = vg.name
    m.invert_vertex_group = True          # seam verts weight 0 -> kept
    m.vertex_group_factor = 10.0
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.modifier_apply(modifier=m.name)
    if o.vertex_groups.get("uv_seams"):
        o.vertex_groups.remove(o.vertex_groups["uv_seams"])


decimate(obj, TARGET_FACES)
print("after decimate", len(me.polygons))

# ---------------------------------------------------------------- nozzles onto the engine ring
co = np.empty(len(me.vertices) * 3)
me.vertices.foreach_get("co", co)
co = co.reshape(-1, 3)
bells = co[co[:, 2] < Z_EXIT + 0.02, :2]
# 9 nozzles (one centre + ring of 8, confirmed from a bottom-up render): k-means seeded on that layout
rr = np.percentile(np.hypot(*bells.T), 75)
best = None
for off in np.linspace(0, math.pi / 4, 9, endpoint=False):
    C = np.array([[0, 0]] + [[rr * math.cos(off + i * math.pi / 4), rr * math.sin(off + i * math.pi / 4)] for i in range(8)])
    for _ in range(30):
        lab = np.argmin(((bells[:, None, :] - C[None]) ** 2).sum(-1), 1)
        C = np.array([bells[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(9)])
    inertia = ((bells - C[lab]) ** 2).sum()
    if best is None or inertia < best[0]:
        best = (inertia, C, lab)
_, C, lab = best
exits = []
for j in range(9):
    rim = co[co[:, 2] < Z_EXIT + 0.004, :2]
    P = rim[np.hypot(*(rim - C[j]).T) < 0.029]   # half the ring pitch
    exits.append((C[j], float(np.percentile(np.hypot(*(P - C[j]).T), 90))))
print("nozzle exits", [(np.round(m, 4).tolist(), round(r, 4)) for m, r in exits])
exits.sort(key=lambda e: np.hypot(*e[0]))
# Keep Meshy's nozzles exactly as modelled: moving or narrowing the bells tears the gimbal actuators.
# The plume still uses the physical exit radius from vehicle.json; the bells are ~20 % wider than that.
engine_positions = [[float(m[0] * K), 0.0, float(-m[1] * K)] for m, _ in exits]  # glTF: x, y(up), z = -blender_y


# ---------------------------------------------------------------- lengthen barrels by repeating bands
def repeat_band(zb, zt, copies):
    bm = bmesh.new()
    bm.from_mesh(me)
    for z in (zb, zt):
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, z), plane_no=(0, 0, 1))
    L = zt - zb - OVERLAP
    bm.faces.ensure_lookup_table()
    above = [f for f in bm.faces if f.calc_center_median().z > zt]
    band = [f for f in bm.faces if zb < f.calc_center_median().z < zt]
    res = bmesh.ops.split(bm, geom=above)
    moved = {v for f in res["geom"] if isinstance(f, bmesh.types.BMFace) for v in f.verts}
    bmesh.ops.translate(bm, verts=list(moved), vec=(0, 0, copies * L))
    for i in range(1, copies + 1):
        dup = bmesh.ops.duplicate(bm, geom=band)
        dv = [g for g in dup["geom"] if isinstance(g, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, verts=dv, vec=(0, 0, i * L))
    bm.to_mesh(me)
    bm.free()
    me.update()
    return copies * L


d2 = repeat_band(*S2_BAND, S2_COPIES)          # upper band first, in original coordinates
d1 = repeat_band(*S1_BAND, S1_COPIES)
z_inter0 = Z_INTER0 + d1
z_inter1 = Z_INTER1 + d1
z_fair = Z_FAIR + d1 + d2
z_flare = Z_FLARE + d1 + d2
z_nose = float(np.max([v.co.z for v in me.vertices]))
print("post-repeat landmarks", z_inter0, z_inter1, z_fair, z_nose)

# ---------------------------------------------------------------- materials: clean interstage
src_mat = me.materials[0]
inter_mat = bpy.data.materials.new("Interstage")
inter_mat.use_nodes = True
bsdf = inter_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.018, 0.019, 0.021, 1)
bsdf.inputs["Roughness"].default_value = 0.55
bsdf.inputs["Metallic"].default_value = 0.25
inter_mat.use_backface_culling = False
me.materials.append(inter_mat)

# split cleanly at the stage boundaries and the fairing seam
bm = bmesh.new()
bm.from_mesh(me)
for z in (z_inter1, z_fair):
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, z), plane_no=(0, 0, 1))
bm.faces.ensure_lookup_table()
fair = [f for f in bm.faces if f.calc_center_median().z > z_fair]
geom = list({v for f in fair for v in f.verts}) + list({e for f in fair for e in f.edges}) + fair
bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(X_SEAM, 0, 0), plane_no=(1, 0, 0))
for f in bm.faces:
    c = f.calc_center_median()
    if z_inter0 < c.z < z_inter1:
        f.material_index = 1
bm.to_mesh(me)
bm.free()


# ---------------------------------------------------------------- piecewise map to vehicle metres
def ymap(z):
    zs = [Z_EXIT, z_inter0, z_inter1, z_fair, z_nose]
    ys = [0.0, Y_INTER0, Y_INTER1, Y_FAIR, Y_NOSE]
    # nozzle and engine section keep their true size (scale K); the tank absorbs the rest
    z_eng = Z_TANK0
    if z <= z_eng:
        return (z - Z_EXIT) * K
    zs[0], ys[0] = z_eng, (z_eng - Z_EXIT) * K
    return float(np.interp(z, zs, ys))


fs = FAIR_R / (0.095 * K)   # widen fairing to 2.6 m
co = np.empty(len(me.vertices) * 3)
me.vertices.foreach_get("co", co)
co = co.reshape(-1, 3)
y = np.array([ymap(z) for z in co[:, 2]])
g = np.clip((co[:, 2] - z_fair) / (z_flare - z_fair), 0, 1)
xy_scale = K * (1 + (fs - 1) * g)
co[:, 0] *= xy_scale
co[:, 1] *= xy_scale
co[:, 2] = y
me.vertices.foreach_set("co", co.ravel())
me.update()
print("fairing radial scale", round(fs, 3), "tank stretch", round((Y_INTER0 - (Z_TANK0 - Z_EXIT) * K) / ((z_inter0 - Z_TANK0) * K), 3))

# ---------------------------------------------------------------- separate parts
def separate(name, pred):
    bpy.ops.object.select_all(action="DESELECT")
    src = bpy.data.objects[obj.name]
    bpy.context.view_layer.objects.active = src
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for p in src.data.polygons:
        p.select = pred(p.center)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    new = [o for o in bpy.context.selected_objects if o != src][0]
    new.name = new.data.name = name
    return new


xs = X_SEAM * K
separate("FairingL", lambda c: c.z > Y_FAIR and c.x >= xs)
separate("FairingR", lambda c: c.z > Y_FAIR and c.x < xs)
separate("S2", lambda c: Y_INTER1 < c.z <= Y_FAIR)
noz = separate("S1_Nozzles", lambda c: c.z < (Z_NOZ - Z_EXIT) * K)
obj.name = obj.data.name = "S1"
noz["engine_positions"] = json.dumps(engine_positions)
noz["exit_radius"] = EXIT_R

for o in bpy.data.objects:
    if o.type == "MESH":
        o.data.materials[:]  # keep slots
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.shade_smooth(keep_sharp_edges=True)
        if o.name in PART_FACES:
            decimate(o, PART_FACES[o.name])
        print("part", o.name, "faces", len(o.data.polygons))

# ---------------------------------------------------------------- textures: keep 2K, WebP (EXT_texture_webp)
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", export_extras=True, export_yup=True,
                          export_image_format="WEBP", export_image_quality=80, export_apply=True)
print("wrote", OUT)
