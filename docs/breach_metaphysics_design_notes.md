# Breach — Metaphysics & "Magic" Design Notes

**Status:** Idea collection. *Not* an implementation spec.
**Purpose:** A rich foundation for systems that will grow over time. The `Unit` class
already carries the *fields* these ideas need (`imagination`, `will_orientation`,
`awakened`); this document holds the *thinking* about what they will eventually drive.
Nothing here is built tonight.

**Source / inspiration:** Franz Hartmann, *Magic, White and Black* (1886) — a
Theosophical-Hermetic treatise. Reached via the Spanish-language *Magia Negra, Magia
Blanca*. Originally developed for the Opera RPG ("The Treasure at the End of the
Rainbow Opera"); now being ported into Breach, where the Grays make it sit naturally.

---

## 1. The core idea

Hartmann frames white vs. black magic not as two kinds of power but as one power with
two **orientations of the will**:

- **White** — will oriented toward the universal Life; cohesion; reality.
- **Black** — will oriented toward the separative, egoic self; isolation; illusion.

Two faculties are operative:

- **Imagination** (Paracelsian *Imaginatio*) — the image-forming faculty. Every image
  dwelt on is a "seed."
- **Will** — the force that *realises* the image. Its **orientation** is what makes the
  same act white or black.

The metaphysics applies **universally to every being** — humans, xenos, Grays, animals,
robots-with-minds. Alignment is not a chosen label; it is an *emergent state of the
soul* derived from cumulative orientation. The **Mirror People** are not a special case
— they are the extreme black pole of a spectrum everyone is on: black magicians who
went so deep into self-isolating illusion they severed all anchor to reality, now able
to act only through reflections and dreams.

---

## 2. How this maps onto Breach's stats

| Concept | Stat | Player sees it? |
|---|---|---|
| Image-forming faculty | `imagination` | Hidden until awakening |
| The realising force (magnitude) | `will_strength` — shown as **WILL** | Yes |
| Orientation of will (separative ↔ universal) | `will_orientation`, `[-1,+1]` | No (hidden, permanently for now) |

`will_strength` does double duty: it carries familiar RPG mechanics (spell-interrupt
resistance, morale) **on top of** the Hartmann metaphysics. Players recognise WILL from
other RPGs; the metaphysical layer rides underneath it, invisible.

`imagination` exists in the class for every unit always — it is hidden by UI policy,
revealed only when a unit is `awakened`. `will_orientation` is the hidden "vector" half
of will, not shown to the player at all.

---

## 3. The Grays as the black pole

The Grays are a covert, infiltrating, reality-distorting faction — Hartmann's "severance
from the real" wearing a sci-fi costume. They need no metaphysical reskin; they *are*
the black-magic pole expressed as an enemy.

Established Gray capabilities relevant here:
- Energy weapons (conventional combat).
- Mind control.
- **Image seeding** — planting an image rather than dealing direct mind damage.

This makes the Gray threat *structurally* different from the xenos: xenos are a
physical-horror clock; the Grays attack the unit's relationship to reality itself.

---

## 4. System sketches — future, not built

### 4.1 Fear as operative imagination

Hartmann's strongest line: fear is the most powerful imagination most people ever wield,
and almost always against themselves.

Proposed split (Erik's refinement):
- **`will_strength` sets susceptibility** — how easily fear takes hold. Low will
  strength = more easily affected.
- **`will_orientation` sets content** — *what* the unit fears. A separative-oriented
  unit fears loss of self, isolation, its own death. A universal-oriented unit fears
  failing the group, the loss of others.
- **`imagination` sets magnitude** — how vividly the feared image is realised. High
  imagination + weak/misoriented will = the unit conjures threats out of nothing
  (fires at shadows, behaves as if a distant xeno is adjacent, breaks formation). The
  *same* high imagination with strong, well-oriented will = accurate threat
  anticipation. Same faculty, opposite outcome.

This pairs well with Breach's tick-based horror pacing — panic could cost a unit ticks
beyond its baseline cadence.

### 4.2 The Gray hook

The Grays plant a **seed image**; they do not deal direct mind damage. Sketch: each
tick the seed grows by `imagination` and is suppressed by well-oriented `will`. If it
crosses a threshold, the unit manifests the false reality — sees allies as enemies,
believes a safe room is breached. This makes infiltration *mechanical*, and makes the
player genuinely want strong, well-oriented units near Grays — a meaningful constraint,
not just difficulty. (Design principle: "Difficulty isn't what makes combat good —
meaningful constraint does.")

### 4.3 Will-orientation drift — **open, deliberately unsolved**

Orientation could drift over a campaign: cohesion acts (covering squadmates, holding
the line) pull toward universal; isolating acts (breaking formation, surviving at
others' expense) pull separative.

**Caution.** Most games that try to quantise morality fail at it. The bar here is high.
Do **not** build this until the normal stats are solid and there is a genuinely good
mechanic — a vague good/evil slider is worse than nothing. Treat it as an open design
problem, not a scheduled feature. Drift also only pays off once Breach has a campaign
meta-layer (a prototype mission exists; a campaign and several mission ideas are
planned).

### 4.4 Manifestation — the white pole

The mirror image of the Gray hook. A unit far toward universal-oriented will with high
imagination can *manifest* small reality effects — steadying a decompressing room
against entropy, sensing a hidden Gray. Keep it rare, costly, and opaque (Dark
Souls-style discovery; no wiki-readable transparency).

### 4.5 The awakening trigger

`imagination` is hidden on the character sheet until the unit is `awakened`. Awakening
is earned by surviving a **"breach of the real"** — direct Gray contact, a true-death
save (the `Downed` life-state survived), or witnessing a Mirror Person. The "breach"
metaphor is already in the project's DNA — and in its name.

Design intent: it is fine — desirable, even — if most players never awaken a unit and
never see the `imagination` stat exist. Hidden content changes how the visible content
*feels* for everyone; the few who uncover it feel they discovered something real about
the world. (Opacity principle, carried over from the Opera RPG.)

---

## 5. The Mirror People

The extreme black pole, expressed as entities/enemies:
- Former black magicians whose will became so oriented toward the separative self that
  the egoic structure severed their access to reality.
- They exist only in reflections — mirrors, still water — and in dreams. They can see
  and act through reflective/dreamlike surfaces but cannot enter the real.
- Mechanically: `will_orientation` pinned near `-1`, both `will_strength` and
  `imagination` high. They are a walking demonstration of Hartmann's thesis — not an
  exception to the world's metaphysics but its most extreme instance.

---

## 6. Summary of what is decided vs. open

**Decided:**
- `imagination`, `will_strength`, `will_orientation`, `awakened` are fields on `Unit`.
- Player sees WILL (`will_strength`) only; `imagination` revealed on awakening;
  `will_orientation` hidden permanently for now.
- Metaphysics applies universally to all beings; Mirror People are the extreme pole.
- Awakening = surviving a "breach of the real."

**Open / deliberately deferred:**
- Fear system (susceptibility / content / magnitude split sketched, not built).
- The Gray hook mechanic.
- Will-orientation drift — only with a genuinely good mechanic and a campaign layer.
- Manifestation / white-pole effects.
- Exact awakening-trigger conditions and what awakening unlocks mechanically.
