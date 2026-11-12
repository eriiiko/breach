"""#29 P3e -- the match path's named sites carry no float arithmetic.

The re-inventoried site table (``docs/residency_batching_design_2026-11-07.md``
§7, "P3e site table, re-inventoried 2026-11-12") names the functions P3e made
integer. This scan parses each and refuses, inside its body:

* a ``math.*`` call (sqrt, radians, floor, ...),
* a ``float(...)`` call,
* a generator ``.uniform`` / ``.random`` / ``.normal`` / ``.standard_normal``
  call (a match draw goes through ``simulation.draws``),
* a true division ``a / b`` (``//`` is integer),
* a float literal used in arithmetic (``0.5 * x``),

unless a line of its statement carries ``p3e-door`` or ``p3e-view`` (a config
constant quantized once / a render or observation float VIEW of an integer)
-- the ``ingress-exempt:`` precedent.

PROPERTY: every listed function is float-free bar its marked doors and views.
BREAKS IF: a float operation returns to a listed site (e.g. ``rng.uniform``
in the shot cone, ``math.sqrt`` in the blast falloff, ``amount * weight`` on
floats in the FieldEdit combine) without a marked reason. (Validated by
breaking it: a ``math.sqrt`` put back into ``apply_blast_damage`` and a
``float(`` into ``BulletInFlight.advance`` each turn it red.)
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "src" / "simulation"

#: (module file, qualified function) -- the P3e sites, by group.
SITES = [
    # (a) the shot cone and the line march
    ("combat.py", "BulletInFlight.advance"),
    ("combat.py", "fire_burst"),
    ("combat.py", "fire_beam"),
    ("combat.py", "_directional_fire"),
    # (b) the rolls
    ("attack_resolver.py", "cover_exposure_at"),
    ("attack_resolver.py", "roll_exposure"),
    ("attack_resolver.py", "arc_multiplier"),
    ("attack_resolver.py", "roll_crit"),
    ("draws.py", "cone_offset_q16"),
    ("draws.py", "roll_q16"),
    ("draws.py", "noise_mult_q16"),
    # (c) mitigation and the hp
    ("damage.py", "mitigate_q"),
    ("damage.py", "apply_packet"),
    ("damage.py", "DamagePacket.q"),
    ("status.py", "tick_statuses"),
    # (g) the heat chain and the poison dose
    ("exchange.py", "heat_dials_q"),
    ("exchange.py", "felt_temperature_q"),
    ("exchange.py", "apply_environmental_damage"),
    ("exchange.py", "apply_poison_dose"),
    # (f) FieldEdit and the stamp refill
    ("field_edit.py", "_iter_region"),
    ("field_edit.py", "apply_field_edit"),
    ("field_edit.py", "_combine_q"),
    ("field_edit.py", "_combine_heat_q"),
    ("field_edit.py", "amount_q_of"),
    ("gamemap.py", "GameMap._neighbor_mean"),
    # (e) blasts, payloads, sprays, grenade flight
    ("physics.py", "apply_explosion"),
    ("exchange.py", "apply_blast_damage"),
    ("payloads.py", "linear_weights"),
    ("payloads.py", "deposit_over_tiles"),
    ("payloads.py", "blast_smoke_peak_q"),
    ("payloads.py", "execute_payload"),
    ("blast_expansion.py", "_source_q"),
    ("heat_source_system.py", "_blast_damage_q16"),
    ("combat.py", "Projectile.update_position"),
    ("combat.py", "spray_cone_tiles"),
    ("combat.py", "deposit_spray_cone"),
    ("combat.py", "_enqueue_spray_cells"),
    ("combat.py", "start_spray_burst_directional"),
    ("simulation.py", "Simulation._release_grenade"),
]

_DRAW_NAMES = {"uniform", "random", "normal", "standard_normal"}
_MARKS = ("p3e-door", "p3e-view")


def _find(tree, qual):
    parts = qual.split(".")
    nodes = tree.body
    found = None
    for i, name in enumerate(parts):
        found = None
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
                found = n
                break
        assert found is not None, f"{qual}: {name} not found"
        nodes = found.body
    return found


def _marked_lines(fn, lines):
    """Every line of a statement carrying a mark: a simple statement's whole
    span, a compound one's header (its lines before its body)."""
    marked = set()
    for st in ast.walk(fn):
        if not isinstance(st, ast.stmt) or st is fn:
            continue
        body = getattr(st, "body", None)
        end = (body[0].lineno - 1 if isinstance(body, list) and body
               else st.end_lineno)
        span = range(st.lineno, max(end, st.lineno) + 1)
        if any(m in lines[i - 1] for i in span for m in _MARKS):
            marked.update(span)
    return marked


def _offences(fn, lines):
    out = []
    marked = _marked_lines(fn, lines)
    for node in ast.walk(fn):
        line = getattr(node, "lineno", None)
        if line is None:
            continue
        text = lines[line - 1]
        if line in marked:
            continue
        bad = None
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and f.value.id == "math" and f.attr != "isqrt"):
                bad = f"math.{f.attr}"
            elif isinstance(f, ast.Name) and f.id == "float":
                bad = "float()"
            elif isinstance(f, ast.Attribute) and f.attr in _DRAW_NAMES:
                bad = f".{f.attr}()"
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Div):
                bad = "true division"
            elif any(isinstance(x, ast.Constant) and isinstance(x.value, float)
                     for x in (node.left, node.right)):
                bad = "float literal in arithmetic"
        if bad:
            out.append(f"line {line}: {bad}: {text.strip()}")
    return out


@pytest.mark.parametrize("module,qual", SITES,
                         ids=[f"{m}::{q}" for m, q in SITES])
def test_site_is_float_free(module, qual):
    src = (SIM / module).read_text(encoding="utf-8")
    fn = _find(ast.parse(src), qual)
    off = _offences(fn, src.splitlines())
    assert not off, f"{module}::{qual} has float sites:\n" + "\n".join(off)
