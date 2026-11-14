"""Verify the editable saved review file, not runtime/rig compatibility."""
import bpy
import json
from pathlib import Path
from mathutils import Vector

scene=bpy.context.scene
expected=['AGENT | unrigged silhouette surfaces','EQUIPMENT | removable soft duffel placeholder','EQUIPMENT | removable compact weapon placeholder']
assert all(bpy.data.collections.get(n) for n in expected), 'Independent asset collections missing'
assert all(any(o.type=='MESH' for o in bpy.data.collections[n].objects) for n in expected)
assert not any(o.type=='ARMATURE' for o in scene.objects), 'Rigging unexpectedly introduced'
camera=bpy.data.objects['REVIEW top_down_vertical']
assert camera.data.type=='ORTHO'
direction=camera.matrix_world.to_quaternion() @ Vector((0,0,-1))
assert (direction-Vector((0,0,-1))).length<1e-6, 'Vertical review camera tilted'
for m in bpy.data.materials:
    if m.use_nodes:
        p=m.node_tree.nodes.get('Principled BSDF')
        if p: assert p.inputs['Emission Strength'].default_value==0, m.name+' has emission'
preview=Path(bpy.data.filepath).parents[1]/'previews'
views=['front','side','back','three_quarter','top_down_vertical','top_down_tilted_readability']
assert all((preview/(v+'.png')).is_file() for v in views), 'Review render missing'
result={'saved_file_reopened':True,'independent_character_bag_weapon':True,'no_rig_introduced':True,'vertical_camera':list(direction),'surface_emission_disabled':True,'review_renders_present':views}
(Path(bpy.data.filepath).parent/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('REOPEN_VERIFIED',json.dumps(result))
