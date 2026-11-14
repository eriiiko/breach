"""Compose the six actual renders into a labelled comparison; no image synthesis."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
FONT=Path("C:/Windows/Fonts/segoeui.ttf")
font=ImageFont.truetype(str(FONT),25)
small=ImageFont.truetype(str(FONT),19)
board=Image.new("RGB",(1880,1860),(233,230,222))
draw=ImageDraw.Draw(board)
draw.text((24,15),"One sleeve / equal camera, lighting and source materials",font=font,fill=(28,32,33))
draw.text((24,50),"Accepted baseline and two bounded studies. Applied geometry; no rollout to the character.",font=small,fill=(45,50,51))
for col,(prefix,label) in enumerate((("baseline","Accepted cleanup baseline"),
                                   ("candidate_1","Candidate 1 / broad compression"),
                                   ("candidate_2","Candidate 2 / localized compression"))):
    x=20+col*620
    for row,view in enumerate(("front","side")):
        y=100+row*875
        draw.text((x,y),label+" / "+view,font=small,fill=(28,32,33))
        with Image.open(ROOT/f"{prefix}_{view}.png") as image:
            image=image.convert("RGB").resize((600,840),Image.Resampling.LANCZOS)
            board.paste(image,(x,y+30))
board.save(ROOT/"comparison.png")
print(ROOT/"comparison.png")
