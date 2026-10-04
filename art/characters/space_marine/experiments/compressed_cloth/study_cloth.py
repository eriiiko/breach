"""Render the cloth study separately; does not touch the deliverable .blend."""
import sys
from pathlib import Path
import bpy

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import build_space_marine as author
from cloth_authoring import compressed_limb,bake_compressed_cloth,sleeve_profiles

author.SOURCE=Path(__file__).resolve().parent
author.PREVIEWS=author.SOURCE

bpy.ops.object.select_all(action="SELECT");bpy.ops.object.delete(use_global=False)
author.make_materials()
obj,meta=compressed_limb(author,"STUDY / continuous compressed sleeve",sleeve_profiles(),[(1.008,1.137),(1.239,1.427)])
bake_compressed_cloth([obj])
sub=obj.modifiers.new("Cloth surface interpolation","SUBSURF");sub.levels=1;sub.render_levels=1
author.setup_studio()
camera=bpy.data.objects["Camera / hero"]
camera.location=(2,-5,2.0);author.aim(camera,(.38,0,1.25));camera.data.ortho_scale=.74
bpy.context.scene.camera=camera
scene=bpy.context.scene;scene.render.resolution_x=1000;scene.render.resolution_y=1400;scene.cycles.samples=64
scene.render.filepath=str(author.PREVIEWS/"draft_cloth_study.png")
bpy.ops.wm.save_as_mainfile(filepath=str(author.SOURCE/"cloth_study.blend"),compress=True)
bpy.ops.render.render(write_still=True)
