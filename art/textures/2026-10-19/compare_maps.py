"""Reproducible Laigter default baseline for three representative materials.

Uses the installed executable without changing its presets or source artwork.
Existing inferred-height normals remain the production candidates. A default
baseline is an experiment, not a claim about the best achievable Laigter result.
"""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from review_maps import ROOT, OUT, FONT, SMALL, light_image, panel

LAIGTER = Path("C:/Users/steen/projects/laigter/dist_win/laigter_app/laigter-windows/laigter.exe")
CASES = ["02-engine-room/realistic/floor.png", "03-wood-quarters/realistic/floor-oak.png",
         "10-washrooms/realistic/floor-hex.png"]


def main():
    report=[]
    for name in CASES:
        path=ROOT/name
        dest=OUT/"laigter-baseline"/Path(name).parent
        dest.mkdir(parents=True,exist_ok=True)
        command=[str(LAIGTER),"--no-gui","-d",str(path),"-n","--flatten","-l",str(dest)]
        r=subprocess.run(command,capture_output=True,text=True,timeout=60)
        if r.returncode:
            raise RuntimeError(r.stderr or r.stdout or str(r.returncode))
        baseline=dest/(path.stem+"_n.png")
        with Image.open(path) as source:
            colour=source.convert("RGBA").resize((320,320),Image.Resampling.LANCZOS)
        chosen=ROOT/"normal-v2"/name
        ours=Image.open(chosen.with_name(path.stem+"_n.png")).resize((320,320),Image.Resampling.BILINEAR)
        theirs=Image.open(baseline).resize((320,320),Image.Resampling.BILINEAR)
        frames=[]
        for f in range(24):
            angle=2*math.pi*f/24
            im=Image.new("RGB",(960,710),(24,26,31))
            draw=ImageDraw.Draw(im)
            draw.text((8,6),name,font=FONT,fill="white")
            for x,title in [(0,"Colour"),(320,"Height -> normal v2"),(640,"Laigter default baseline")]:
                draw.text((x+8,32),title,font=SMALL,fill="white")
            im.paste(panel(colour,320),(0,60))
            im.paste(light_image(ours,angle),(320,60))
            im.paste(light_image(theirs,angle),(640,60))
            im.paste(light_image(ours,angle,colour),(320,390))
            im.paste(light_image(theirs,angle,colour),(640,390))
            draw.text((8,395),"Top: neutral surface",font=SMALL,fill="white")
            draw.text((8,419),"Bottom: colour + same light",font=SMALL,fill="white")
            draw.text((8,453),"Diagnostic light; not game rendering.",font=SMALL,fill="white")
            frames.append(im)
        stem="--".join(Path(name).with_suffix("").parts)
        frames[0].save(OUT/(stem+"--comparison.gif"),save_all=True,append_images=frames[1:],duration=85,loop=0)
        frames[3].save(OUT/(stem+"--comparison.jpg"),quality=95)
        a=np.asarray(Image.open(baseline).convert("RGB"),dtype=np.float32)/127.5-1
        report.append({"asset":name,"command":command,"returncode":r.returncode,
                       "baseline_size":list(Image.open(baseline).size),
                       "green_flip_applied":False,
                       "note":"Installed defaults; no profile tuning. Interpret orientation separately from geometry.",
                       "normal_z_min":float(a[...,2].min())})
        print(name,flush=True)
    (OUT/"laigter-baseline.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
