"""
Export marine sprites from HTML pixel art to PNG files.

Recreates the south-facing marine sprite from the JS drawPixel/drawRect calls
using Pillow, then rotates it for all 8 directions.

Output:
  art/sprites/marine/marine_{dir}.png   — individual 32x32 PNGs
  art/sprites/marine/marine_8dir_sheet.png — 8 sprites in a row
"""

from pathlib import Path
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Color palette (from marine_8dir.html)
# ---------------------------------------------------------------------------
COLORS = {
    'armorDark':      '#252e22',
    'armorMid':       '#2e3b28',
    'armorBase':      '#3a4a32',
    'armorLight':     '#4a5a40',
    'armorHighlight': '#566a4c',
    'spine':          '#354530',
    'belt':           '#1a2117',
    'buckle':         '#3d3520',
    'neck':           '#1e2619',
    'boot':           '#1a2117',
    'bootTop':        '#1e2619',
    'gunDark':        '#181818',
    'gunBody':        '#222222',
    'gunMid':         '#282828',
    'gunLight':       '#333333',
    'gunRail':        '#2a2a2a',
    'gunMetal':       '#2e2e2e',
    'visorDark':      '#0f1510',
    'visorGlass':     '#1a2820',
    'comms':          '#252e22',
    'glove':          '#1e2619',
    'squadMark':      '#556a48',
}

# Unpack for convenience
armorDark     = COLORS['armorDark']
armorMid      = COLORS['armorMid']
armorBase     = COLORS['armorBase']
armorLight    = COLORS['armorLight']
armorHighlight= COLORS['armorHighlight']
spine         = COLORS['spine']
belt          = COLORS['belt']
buckle        = COLORS['buckle']
neck          = COLORS['neck']
boot          = COLORS['boot']
bootTop       = COLORS['bootTop']
gunDark       = COLORS['gunDark']
gunBody       = COLORS['gunBody']
gunMid        = COLORS['gunMid']
gunLight      = COLORS['gunLight']
gunRail       = COLORS['gunRail']
gunMetal      = COLORS['gunMetal']
visorDark     = COLORS['visorDark']
visorGlass    = COLORS['visorGlass']
comms         = COLORS['comms']
glove         = COLORS['glove']
squadMark     = COLORS['squadMark']

# ---------------------------------------------------------------------------
# Drawing helpers (match JS semantics: x, y, w, h in pixels)
# ---------------------------------------------------------------------------

def draw_rect(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, color: str):
    """Fill a rectangle. PIL rectangle uses inclusive coords for both corners."""
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=color)


def draw_pixel(draw: ImageDraw.ImageDraw, x: int, y: int, color: str):
    """Draw a single pixel."""
    draw.point((x, y), fill=color)


# ---------------------------------------------------------------------------
# Build the south-facing marine (exact replica of drawMarineSouth)
# ---------------------------------------------------------------------------

def draw_marine_south() -> Image.Image:
    img = Image.new('RGBA', (32, 32), (0, 0, 0, 0))
    d = img

    ctx = ImageDraw.Draw(d)

    # BOOTS
    draw_rect(ctx, 12, 26, 2, 2, bootTop)
    draw_rect(ctx, 18, 26, 2, 2, bootTop)

    # LEGS
    draw_rect(ctx, 11, 23, 3, 3, armorMid)
    draw_pixel(ctx, 12, 24, spine)
    draw_rect(ctx, 18, 23, 3, 3, armorMid)
    draw_pixel(ctx, 19, 24, spine)

    # TORSO
    draw_rect(ctx, 10, 12, 12, 10, armorBase)
    draw_rect(ctx, 10, 12, 12, 1, armorLight)
    draw_rect(ctx, 10, 21, 12, 1, armorMid)
    draw_rect(ctx, 10, 12, 1, 10, armorMid)
    draw_rect(ctx, 21, 12, 1, 10, armorMid)
    draw_rect(ctx, 15, 11, 2, 13, spine)
    draw_rect(ctx, 11, 14, 3, 6, armorLight)
    draw_rect(ctx, 18, 14, 3, 6, armorLight)

    # BELT
    draw_rect(ctx, 10, 21, 12, 2, belt)
    draw_rect(ctx, 10, 21, 2, 2, armorDark)
    draw_rect(ctx, 20, 21, 2, 2, armorDark)

    # LIFE SUPPORT PACK
    draw_rect(ctx, 12, 14, 8, 6, armorDark)
    draw_rect(ctx, 13, 15, 6, 4, neck)
    draw_rect(ctx, 12, 14, 8, 1, armorMid)
    draw_rect(ctx, 13, 15, 2, 3, spine)
    draw_pixel(ctx, 14, 16, armorLight)
    draw_rect(ctx, 17, 15, 2, 3, spine)
    draw_pixel(ctx, 18, 16, armorLight)
    draw_rect(ctx, 15, 16, 2, 2, belt)
    draw_pixel(ctx, 15, 13, neck)

    # SHOULDER PADS - Left
    draw_rect(ctx, 7, 12, 4, 7, armorBase)
    draw_rect(ctx, 7, 12, 4, 1, armorHighlight)
    draw_rect(ctx, 7, 18, 4, 1, armorMid)
    draw_rect(ctx, 7, 13, 1, 5, armorLight)
    draw_rect(ctx, 8, 14, 2, 4, armorLight)
    # Right
    draw_rect(ctx, 21, 12, 4, 7, armorBase)
    draw_rect(ctx, 21, 12, 4, 1, armorHighlight)
    draw_rect(ctx, 21, 18, 4, 1, armorMid)
    draw_rect(ctx, 24, 13, 1, 5, armorLight)
    draw_rect(ctx, 22, 14, 2, 4, armorLight)
    draw_rect(ctx, 22, 14, 2, 1, squadMark)
    draw_pixel(ctx, 22, 15, squadMark)

    # ARMS - Left
    draw_rect(ctx, 7, 19, 2, 3, armorMid)
    draw_rect(ctx, 6, 17, 2, 3, armorMid)
    draw_pixel(ctx, 6, 17, armorDark)
    draw_rect(ctx, 4, 15, 2, 3, armorMid)
    draw_rect(ctx, 4, 14, 2, 2, glove)
    # Right
    draw_rect(ctx, 23, 19, 2, 3, armorMid)
    draw_rect(ctx, 24, 17, 2, 3, armorMid)
    draw_rect(ctx, 26, 16, 2, 2, armorMid)

    # WEAPON
    draw_rect(ctx, 4, 12, 3, 2, gunDark)
    draw_rect(ctx, 3, 14, 4, 4, gunBody)
    draw_rect(ctx, 3, 14, 4, 1, gunMetal)
    draw_rect(ctx, 4, 14, 2, 4, gunRail)
    draw_rect(ctx, 3, 16, 1, 2, gunDark)
    draw_rect(ctx, 4, 18, 2, 4, gunMid)
    draw_rect(ctx, 4, 22, 2, 2, gunLight)
    draw_pixel(ctx, 5, 23, gunBody)
    draw_rect(ctx, 4, 17, 2, 1, gunMetal)
    draw_pixel(ctx, 5, 17, armorDark)

    # HELMET
    draw_rect(ctx, 11, 5, 10, 8, armorBase)
    draw_rect(ctx, 12, 12, 8, 1, armorMid)
    draw_rect(ctx, 12, 4, 8, 1, armorBase)
    draw_rect(ctx, 13, 3, 6, 1, armorLight)
    draw_rect(ctx, 14, 2, 4, 1, armorLight)
    draw_rect(ctx, 15, 2, 2, 11, spine)
    draw_rect(ctx, 15, 3, 2, 2, armorLight)
    draw_rect(ctx, 13, 4, 2, 3, armorLight)
    draw_pixel(ctx, 14, 3, armorHighlight)
    draw_rect(ctx, 11, 6, 1, 6, armorMid)
    draw_rect(ctx, 20, 6, 1, 6, armorMid)
    draw_rect(ctx, 10, 7, 1, 3, comms)
    draw_rect(ctx, 21, 7, 1, 3, comms)
    draw_rect(ctx, 12, 11, 3, 1, visorDark)
    draw_rect(ctx, 17, 11, 3, 1, visorDark)
    draw_rect(ctx, 14, 12, 4, 1, visorGlass)
    draw_rect(ctx, 12, 12, 8, 1, neck)

    return img


# ---------------------------------------------------------------------------
# Rotation angles: PIL rotates counter-clockwise, so we use positive angles
# S is the base (0°). Going clockwise: SW=45 CCW, W=90 CCW, etc.
# ---------------------------------------------------------------------------
DIRECTIONS = {
    'S':  0,
    'SW': 45,
    'W':  90,
    'NW': 135,
    'N':  180,
    'NE': 225,
    'E':  270,
    'SE': 315,
}

# Canonical order for the sprite sheet
DIR_ORDER = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


def main():
    out_dir = Path(r'C:\Users\steen\projects\breach\art\sprites\marine')
    out_dir.mkdir(parents=True, exist_ok=True)

    south = draw_marine_south()

    sprites = {}
    for name, angle in DIRECTIONS.items():
        if angle == 0:
            sprites[name] = south.copy()
        else:
            sprites[name] = south.rotate(angle, resample=Image.NEAREST, expand=False)

    # Save individual PNGs
    for name, img in sprites.items():
        path = out_dir / f'marine_{name}.png'
        img.save(path)
        print(f'  Saved {path}')

    # Build sprite sheet (8 in a row)
    sheet = Image.new('RGBA', (32 * 8, 32), (0, 0, 0, 0))
    for i, name in enumerate(DIR_ORDER):
        sheet.paste(sprites[name], (i * 32, 0))

    sheet_path = out_dir / 'marine_8dir_sheet.png'
    sheet.save(sheet_path)
    print(f'  Saved {sheet_path}')

    print(f'\nDone — {len(sprites)} directions + 1 sprite sheet.')


if __name__ == '__main__':
    main()
