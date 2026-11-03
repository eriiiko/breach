import bpy, sys
a = sys.argv[sys.argv.index("--")+1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a[0])
for img in bpy.data.images:
    print("IMG", img.name, img.size[:], img.colorspace_settings.name)
    img.filepath_raw = a[1] + "/" + img.name.replace(" ", "_") + ".png"
    img.file_format = "PNG"
    img.save()
for m in bpy.data.materials:
    if m.node_tree:
        for n in m.node_tree.nodes:
            if n.type == "TEX_IMAGE":
                links = [l.to_socket.name for l in n.outputs[0].links]
                print("TEX", m.name, n.image.name, "->", links)
