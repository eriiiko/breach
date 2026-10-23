"""Compose actual renders and labelled crops of Erik's supplied reference."""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT.parents[1]
PREVIEWS=ROOT/"previews"
FINAL=PREVIEWS/"final"
font_path="C:/Windows/Fonts/segoeui.ttf"
font=ImageFont.truetype(font_path,27)
small=ImageFont.truetype(font_path,20)

board=Image.new("RGB",(2480,1360),(233,230,222));draw=ImageDraw.Draw(board)
draw.text((22,14),"Space Marine (Codex) / approved original and additional head variant",font=font,fill=(30,35,36))
draw.text((22,53),"Actual Blender renders: identical cameras, studio lighting and body. Head geometry/material changes only.",font=small,fill=(45,50,51))
for row,(label,prefix) in enumerate((("Approved v1","approved_v1"),("Additional head v2","head_v2"))):
    for col,view in enumerate(("front","side","back","hero")):
        x=20+col*615;y=96+row*630
        draw.text((x,y),label+" / "+view,font=small,fill=(30,35,36))
        with Image.open(FINAL/f"{prefix}_{view}.png") as image:
            board.paste(image.convert("RGB").resize((595,595),Image.Resampling.LANCZOS),(x,y+30))
board.save(PREVIEWS/"comparison_heads.png")

reference=ASSET.parent/"Space Marine Turnaround Sheet.png"
strip=Image.new("RGB",(1120,390),(233,230,222));draw=ImageDraw.Draw(strip)
draw.text((20,12),"Reference details / supplied artwork, cropped and enlarged",font=small,fill=(30,35,36))
with Image.open(reference) as original:
    for index,(name,box) in enumerate((("Front",(242,40,402,190)),
                                     ("Side",(700,40,880,198)),
                                     ("Back",(1133,40,1304,195)))):
        piece=original.crop(box).convert("RGB")
        scale=min(350/piece.width,300/piece.height)
        piece=piece.resize((round(piece.width*scale),round(piece.height*scale)),Image.Resampling.LANCZOS)
        strip.paste(piece,(20+index*366,72))
        draw.text((20+index*366,44),name,font=small,fill=(30,35,36))
strip.save(PREVIEWS/"reference_heads.png")

rows=[]
for view in ("front","side","back","hero"):
    cells=[]
    for prefix,label in (("approved_v1","Approved v1"),("head_v2","Additional head v2")):
        path=f"final/{prefix}_{view}.png"
        cells.append(f'<td><a href="{path}"><img src="{path}" alt="{label}, {view}"></a><p>{label} / {view}</p></td>')
    rows.append("<tr>"+"".join(cells)+"</tr>")
html='''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Space Marine (Codex): head comparison</title>
<style>body{background:#ece9e2;color:#263032;font:17px/1.5 system-ui,sans-serif;max-width:1500px;margin:28px auto;padding:0 20px}a{color:#215979}h1{font-size:30px}img{display:block;width:100%;height:auto}table{width:100%;table-layout:fixed;border-collapse:collapse}th,td{text-align:left;vertical-align:top;padding:8px}td p{margin:5px 0 20px}.reference{max-width:1120px}.hero{max-width:800px}small{font-size:14px}</style>
<h1>Space Marine (Codex): approved original + head v2</h1>
<p>The approved original is preserved. Head v2 adds a shaped side-wrapped visor, broader chin frame and segmented helmet shell. Body, collar and pose are unchanged. Parent visual review is complete; this remains an additional version for Erik's choice.</p>
<p><a href="../../../versions/approved_v1/space_marine_codex_approved_v1.blend">Approved v1 Blender snapshot</a> · <a href="../source/space_marine_codex_head_v2.blend">Head v2 Blender source</a> · <a href="../../../previews/index.html">Original seven-view gallery</a> · <a href="../README.md">Handoff and reproduction</a></p>
<p><a href="comparison_heads.png">Head comparison sheet</a> · <a href="../source/validation.json">Reopened validation</a> · <a href="../../../versions/approved_v1/manifest.json">Original preservation hashes</a> · <a href="../../../../Space%20Marine%20Turnaround%20Sheet.png">Complete supplied reference</a></p>
<a class="reference" href="reference_heads.png"><img src="reference_heads.png" alt="Front, side and back crops from the supplied concept artwork"></a>
<p>Matched neutral close-ups: Cycles, 144 samples, 1600 x 1600. Click an image for full resolution. Helmet construction is still simpler and cleaner than the reference. Cloth and other body limitations are retained from the approved source.</p>
<table><tr><th>Approved v1 / preserved</th><th>Head v2 / additional variant</th></tr>
'''+"\n".join(rows)+'''
</table>
<p>Head v2 full figure, 1600 x 2000. Total evaluated triangles: 1,262,274.</p>
<a class="hero" href="final/head_v2_full_hero.png"><img src="final/head_v2_full_hero.png" alt="Complete Space Marine (Codex), additional head v2"></a>
<p><small>The comparison uses real geometry renders. The reference above is a labelled crop of supplied artwork. No generated or retouched model preview was used. No rigging, UV/game preparation or runtime changes are part of this delivery.</small></p>
</html>
'''
(PREVIEWS/"index.html").write_text(html,encoding="utf-8")

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links.extend(value for key,value in attrs if key in ("href","src"))

links=Links();links.feed(html)
missing=[value for value in links.links if not (PREVIEWS/unquote(value)).exists()]
assert not missing,missing
renders=json.loads((FINAL/"render_manifest.json").read_text(encoding="utf-8"))
conditions=("samples","camera_location","camera_rotation","ortho_scale","resolution")
for view in ("front","side","back","hero"):
    pair=[r for r in renders if r["view"]==view]
    assert len(pair)==2 and all(pair[0][key]==pair[1][key] for key in conditions)
for record in renders:
    path=(ASSET/"versions/approved_v1/space_marine_codex_approved_v1.blend"
          if record["source"]=="approved_v1" else ROOT/"source/space_marine_codex_head_v2.blend")
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record["source_sha256"]
report={"gallery_links_checked":len(links.links),"missing_links":missing,
        "matched_camera_and_render_conditions":True,"render_source_hashes_match":True,
        "reference_sha256":hashlib.sha256(reference.read_bytes()).hexdigest(),
        "reference_image_operation":"Labelled crops and enlargement only; no generative alteration",
        "final_render_count":len(renders)}
(PREVIEWS/"package_validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
