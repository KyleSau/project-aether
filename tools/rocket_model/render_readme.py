"""README hero render of the processed vehicle model: assembled stack beside an exploded view.

    blender -b --factory-startup -P tools/rocket_model/render_readme.py -- frontend/public/models/rocket.glb docs/img/vehicle_model.png
"""
import math
import sys

import bpy

IN, OUT = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
scn.render.engine = "BLENDER_EEVEE"
scn.render.resolution_x, scn.render.resolution_y = 1200, 1400
scn.view_settings.view_transform = "AgX"

world = bpy.data.worlds.new("w")
scn.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.86, 0.87, 0.88, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.9
for rot, energy in [((0.9, 0.0, 0.7), 3.5), ((1.2, 0.0, -2.2), 1.2)]:
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = energy
    sun.angle = 0.05
    o = bpy.data.objects.new("sun", sun)
    o.rotation_euler = rot
    scn.collection.objects.link(o)


def load(offset_x, explode):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=IN)
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        if o.parent is None:
            o.location.x += offset_x
        name = o.name.split(".")[0]
        if explode:
            if name == "S2":
                o.location.z += 4
            elif name == "FairingL":
                o.location.z += 8
                o.location.x += 3.5
            elif name == "FairingR":
                o.location.z += 8
                o.location.x -= 3.5


load(-9, False)
load(9, True)

cam = bpy.data.cameras.new("cam")
cam.lens = 85
c = bpy.data.objects.new("cam", cam)
scn.collection.objects.link(c)
c.location = (0, -185, 43)
c.rotation_euler = (math.radians(88), 0, 0)
scn.camera = c
cam.shift_y = 0.02
cam.lens = 70
scn.render.film_transparent = False
scn.render.filepath = OUT
bpy.ops.render.render(write_still=True)
