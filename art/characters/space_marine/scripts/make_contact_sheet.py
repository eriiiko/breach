"""Assemble labelled, otherwise unaltered model renders. Requires Pillow."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

root=Path(__file__).resolve().parents[1]
preview=root/"previews"
width=2240;margin=32;gap=20;cell=(width-2*margin-3*gap)//4
height=930
canvas=Image.new("RGB",(width,height),(26,33,36))
draw=ImageDraw.Draw(canvas)
font_path=Path("C:/Windows/Fonts/segoeui.ttf")
font=ImageFont.truetype(str(font_path),28) if font_path.is_file() else ImageFont.load_default()
small=ImageFont.truetype(str(font_path),19) if font_path.is_file() else ImageFont.load_default()
draw.text((margin,26),"SPACE MARINE  /  ACTUAL BLENDER MODEL RENDERS",font=font,fill=(235,232,221))
draw.text((margin,71),"Cleanup baseline · 4 October 2026 · Dense editable source · Rigging and game preparation deferred",font=small,fill=(180,196,200))
for i,name in enumerate(("front","side","back","hero")):
    image=Image.open(preview/(name+".png")).convert("RGB")
    image.thumbnail((cell,710),Image.Resampling.LANCZOS)
    x=margin+i*(cell+gap)+(cell-image.width)//2;y=120
    canvas.paste(image,(x,y))
    draw.text((margin+i*(cell+gap),846),name.upper(),font=font,fill=(225,228,220))
draw.text((margin,895),"All views show the same saved .blend. See index.html for the supplied concept, full-size views and source notes.",font=small,fill=(172,188,190))
canvas.save(preview/"contact_sheet.png")
print(preview/"contact_sheet.png")
