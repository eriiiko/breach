"""Beatrice's looks on an opened scene (B2b): the glue between her tables (`LOOKS`, the head), `animehead` and the
generic `toonlook`. Used by `suitbuild` (turnaround, beauty and dev views), `suitpose` (the posed stills) and the face
pictures (`scripts/face_views.py`). A look named `pbr` (or any name whose table entry is None) is the physically-lit
Cycles look of B1a-B2a: nothing is changed.

    look = suitlook.prepare(tables, name, head_name)   # BEFORE posing: custom normals, the hair band, the shadow proxy
    suitlook.apply(tables, look, rig, cam)              # after posing: EEVEE, world, lights, materials, floor, outlines
    suitlook.width(look, cam, target, res)              # per shot: the outline's width in pixels at the target
"""
import animehead
import toonlook


def name_of(tables, name=None):
    """The look's name: `name`, else the tables' default (`LOOK`), else pbr."""
    return name or getattr(tables, "LOOK", "pbr")


def get(tables, name=None):
    return getattr(tables, "LOOKS", {}).get(name_of(tables, name))


def prepare(tables, name=None, head_name=None):
    """Geometry-level preparation of the look on the opened scene (before any posing); returns the look table or None."""
    look = get(tables, name)
    if not look:
        return None
    spec = tables.head(head_name or None)
    if spec.get("kind") != "anime":
        toonlook.prepare(look)
        return look
    head = animehead.Head(spec["face"], spec["eyes"], spec.get("scale", 1.0))
    prefix = tables.PREFIX.capitalize()
    proxies = []
    sp = look.get("shadow_proxy")
    if sp:
        V, Q = head.shadow_proxy(sp.get("h", 0.002), sp.get("shrink", 0.006))
        proxies.append((prefix + "_Head_ShadowProxy", V, Q, sp.get("casters_off", ())))
    normals = {}
    if look.get("face_normals"):
        normals[prefix + "_Head"] = lambda V: head.toon_normals(V, look["face_normals"])
    lines = {}
    if look.get("face_lines"):
        lines[prefix + "_Head"] = lambda V: head.line_mask(V, look["face_lines"])
    toonlook.prepare(look, normals=normals, ring_centre=head.world(spec["face"]["egg"]["centre"]), proxies=proxies,
                     lines=lines)
    return look


def apply(tables, look, rig, cam, backdrop=None):
    if not look:
        return []
    kw = dict(palette=tables.palette())
    if backdrop is not None:
        kw["backdrop"] = backdrop
    return toonlook.apply(look, rig, cam, **kw)


def width(look, cam, target, res):
    """The outlines' width for a shot (call after the camera is placed)."""
    if look:
        import bpy
        bpy.context.view_layer.update()
        toonlook.width_for(cam, target, res, look)


def width_mpp(look, m_per_px, cam=None):
    """The outlines' width for a shot whose pixel size at the subject is known (orthographic, or by distance and lens);
    with `cam` (placed) the view-dependent lines follow its direction."""
    if look and look.get("outline"):
        if cam is not None:
            import bpy
            bpy.context.view_layer.update()
        toonlook.set_width(look["outline"]["px"], m_per_px, cam)
