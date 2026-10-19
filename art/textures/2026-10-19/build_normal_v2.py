"""Build versioned normal/height candidates through the canonical converter.

Colour and generated relief references are preserved. Strengths reproduce the
library's documented per-material artistic starting points. Version 2 fixes
the smoothing precision, not the inferred geometry or physical calibration.
"""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
STRENGTHS={
    "01-clean-quarters": {"realistic":(5,7),"pixel":(4,5),"anime":(4,6)},
    "02-engine-room": {"realistic":(6,8),"anime":(5,7)},
    "03-wood-quarters": {"realistic":(2,3)},
    "04-captains-lodge": {"realistic":(2,5)},
    "05-garden-terrarium": {"realistic":(4,5)},
    "06-aquarium-pool": {"realistic":(3,3)},
    "07-cockpit": {"anime":(4,6)},
    "08-warehouse": {"realistic":(5,8)},
    "09-kitchen": {"realistic":(3,3)},
    "10-washrooms": {"realistic":(3,3)},
    "11-crew-sleeping": {"anime":(1,4)},
}


def main():
    import numpy as np
    from PIL import Image
    rows=[]
    for record in json.loads((ROOT/"validation.json").read_text())["assets"]:
        rel=Path(record["asset"])
        source=ROOT/rel
        rug=source.stem=="rug-persian"
        strength=0.8 if rug else STRENGTHS[rel.parts[0]][rel.parts[1]][int(source.stem.startswith("wall"))]
        target=ROOT/"normal-v2"/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        raw=source.with_name(source.stem+"_height_source.png")
        height=target.with_name(target.stem+"_height.png")
        normal=target.with_name(target.stem+"_n.png")
        command=[sys.executable,str(REPO/"tools/depth_to_normal.py"),str(raw),str(height),str(normal),
                 "--height-input","--strength",str(strength),"--blur","0.8"]
        command += ["--alpha-from",str(source)] if rug else ["--tileable"]
        subprocess.run(command,check=True,capture_output=True)
        with Image.open(source) as image:
            alpha=np.asarray(image.convert("RGBA"))[...,3]
            size=image.size
        with Image.open(height) as image:
            h=np.asarray(image)
            assert image.size==size and h.dtype==np.uint16
        with Image.open(normal) as image:
            assert image.size==size
            encoded=np.asarray(image)
        n=encoded[...,:3].astype(np.float32)/127.5-1
        error=float(np.abs(np.linalg.norm(n,axis=-1)-1).max())
        assert error<0.007 and n[...,2].min()>0
        if rug:
            assert np.array_equal(encoded[...,3],alpha)
            assert np.all(encoded[alpha==0,:3]==(128,128,255))
        rows.append({"colour":rel.as_posix(),"height":height.relative_to(ROOT).as_posix(),
                     "normal":normal.relative_to(ROOT).as_posix(),"strength_pixels":strength,
                     "size":size,"height_unique_levels":int(np.unique(h).size),
                     "max_normal_length_error":error,"wrap_gradients":not rug,
                     "source_alignment_certified":False,"seamless_repeat_certified":False})
        print(rel.as_posix(),flush=True)
    (ROOT/"normal-v2/manifest.json").write_text(json.dumps({"version":2,"assets":rows},indent=2)+"\n")
    lines=["# Normal-map delivery v2 — 2026-10-19", "",
           "29 versioned normal/height pairs, preserving the original colour and relief references.", "",
           "Start with the [visual review](REPORT.md), then browse the surfaces below. Moving-light views are offline diagnostics, not screenshots of Breach.", "",
           "| Surface | Colour | Normal v2 | Height v2 | Moving light | 2x2 repeat |",
           "|---|---|---|---|---|---|"]
    for row in rows:
        rel=Path(row["colour"])
        stem="--".join(rel.with_suffix("").parts)
        normal=Path(row["normal"]).relative_to("normal-v2").as_posix()
        height=Path(row["height"]).relative_to("normal-v2").as_posix()
        lines.append(f"| {rel.with_suffix('').as_posix()} | [colour](../{rel.as_posix()}) | [normal]({normal}) | [height]({height}) | [animation](review/{stem}.gif) | [repeat](review/{stem}--repeat.jpg) |")
    (ROOT/"normal-v2/README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
