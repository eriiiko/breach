"""Textures inside a game .glb, without Blender and without touching anything else in the file.

A game asset is ONE binary glTF: mesh, skin, clips and its two textures (albedo, normal map)
in one buffer. A re-bake of the textures alone (`gameready.py --retexture`, the gloss mask of
#33) must leave the mesh, the skin and the clips exactly as they were, so it never goes back
through an exporter: the new PNG replaces the old one's buffer view here, the other views are
copied byte for byte (only their offsets move), and the JSON keeps every other field.

    python glbtex.py info  <file.glb>              the material's textures and every view's role
    python glbtex.py digest <file.glb>             sha256 of the mesh, skin and animation views
    python glbtex.py compare <a.glb> <b.glb>       those digests side by side (exit 1 if they differ)

Pure Python (json, struct, hashlib): runs in Blender's Python and in the game's.
"""
import hashlib
import json
import struct
import sys

_MAGIC, _JSON, _BIN = 0x46546C67, 0x4E4F534A, 0x004E4942


def read(path):
    """(gltf dict, bin bytes) of a .glb."""
    with open(path, "rb") as f:
        data = f.read()
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    if magic != _MAGIC:
        raise ValueError("%s is not a binary glTF" % path)
    off, gltf, binary = 12, None, b""
    while off < len(data):
        n, kind = struct.unpack_from("<II", data, off)
        chunk = data[off + 8:off + 8 + n]
        if kind == _JSON:
            gltf = json.loads(chunk.decode("utf-8"))
        elif kind == _BIN:
            binary = bytes(chunk)
        off += 8 + n
    return gltf, binary


def write(path, gltf, binary):
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", _MAGIC, 2, total))
        f.write(struct.pack("<II", len(js), _JSON) + js)
        f.write(struct.pack("<II", len(binary), _BIN) + binary)


def view_bytes(gltf, binary, i):
    bv = gltf["bufferViews"][i]
    o = bv.get("byteOffset", 0)
    return binary[o:o + bv["byteLength"]]


def texture_image(gltf, slot, material=0):
    """Image index of a material's texture: slot "baseColor" or "normal"."""
    m = gltf["materials"][material]
    ref = m["pbrMetallicRoughness"]["baseColorTexture"] if slot == "baseColor" else m["normalTexture"]
    return gltf["textures"][ref["index"]]["source"]


def image_png(path, slot):
    """The PNG bytes of a material's texture ("baseColor" or "normal")."""
    g, b = read(path)
    return view_bytes(g, b, g["images"][texture_image(g, slot)]["bufferView"])


def replace_image(path, slot, png, out=None):
    """Replace the PNG of the material's `slot` texture by `png` (bytes); every other buffer view is
    copied unchanged, 4-byte aligned, in its old order. Writes `out` (default: in place)."""
    g, b = read(path)
    target = g["images"][texture_image(g, slot)]["bufferView"]
    order = sorted(range(len(g["bufferViews"])), key=lambda i: g["bufferViews"][i].get("byteOffset", 0))
    parts, pos = [], 0
    for i in order:
        data = png if i == target else view_bytes(g, b, i)
        pad = -pos % 4
        parts.append(b"\0" * pad)
        pos += pad
        bv = g["bufferViews"][i]
        bv["byteOffset"], bv["byteLength"] = pos, len(data)
        parts.append(data)
        pos += len(data)
    binary = b"".join(parts)
    g["buffers"][0]["byteLength"] = len(binary)
    write(out or path, g, binary)


def roles(gltf):
    """bufferView index -> "mesh" | "skin" | "animation" | "image" (by what references it)."""
    acc_role = {}
    for m in gltf.get("meshes", []):
        for p in m["primitives"]:
            for a in list(p["attributes"].values()) + ([p["indices"]] if "indices" in p else []):
                acc_role[a] = "mesh"
    for s in gltf.get("skins", []):
        if "inverseBindMatrices" in s:
            acc_role[s["inverseBindMatrices"]] = "skin"
    for an in gltf.get("animations", []):
        for s in an["samplers"]:
            acc_role[s["input"]] = acc_role[s["output"]] = "animation"
    out = {}
    for a, r in acc_role.items():
        bv = gltf["accessors"][a].get("bufferView")
        if bv is not None:
            out[bv] = r
    for im in gltf.get("images", []):
        out[im["bufferView"]] = "image"
    return out


def digest(path):
    """sha256 per role over the role's buffer views (in index order), plus one over the JSON of
    the meshes, skins, animations, accessors and nodes, and the view count per role."""
    g, b = read(path)
    rl = roles(g)
    out = {}
    for role in ("mesh", "skin", "animation"):
        views = sorted(i for i, r in rl.items() if r == role)
        h = hashlib.sha256()
        for i in views:
            h.update(view_bytes(g, b, i))
        out[role] = dict(views=len(views), bytes=sum(g["bufferViews"][i]["byteLength"] for i in views),
                         sha256=h.hexdigest())
    meta = {k: g.get(k) for k in ("meshes", "skins", "animations", "accessors", "nodes")}
    out["json"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    out["unaccounted_views"] = len(g["bufferViews"]) - len(rl)
    return out


def _main(argv):
    cmd = argv[0]
    if cmd == "info":
        g, _b = read(argv[1])
        print(json.dumps(dict(materials=g["materials"], images=g["images"],
                              base_color_image=texture_image(g, "baseColor"), normal_image=texture_image(g, "normal")),
                         indent=1))
        return 0
    if cmd == "digest":
        print(json.dumps(digest(argv[1]), indent=1))
        return 0
    if cmd == "compare":
        a, b = digest(argv[1]), digest(argv[2])
        same = a == b
        print(json.dumps(dict(same=same, a=a, b=b), indent=1))
        return 0 if same else 1
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
