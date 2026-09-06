"""renderer/propgen.py gate (props & vegetation arc #60 P2).

propgen is the canonical procedural prop/vegetation geometry source. Its
contract:

  * SEEDED-DETERMINISTIC — the same arguments always produce byte-identical
    arrays. Model caching, level reloads and "the tree I placed yesterday"
    all depend on it.
  * A FROZEN public signature: ``generate_tree(seed, palette, height, style,
    decor, decor_density)`` / ``generate_palm(seed, palette, height)``.
  * Triangle arrays: verts/normals float32 (N, 3), colors uint8 (N, 4),
    N % 3 == 0 — and vertex-color ALPHA is the wind-flutter weight (DATA),
    never opacity.
  * Pure: it imports nothing from ``simulation`` (render-only float geometry
    under the Q16.16 render exemption).

No GL context needed — this is numpy in, numpy out.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from renderer import propgen


# ---------------------------------------------------------------------------
# Determinism — THE reason this module exists as a cacheable source
# ---------------------------------------------------------------------------

_TREE_CASES = [
    dict(seed=1, palette="green", height=2.2, style="smooth", decor=None),
    dict(seed=7, palette="exotic", height=3.0, style="smooth",
         decor="flowers"),
    dict(seed=7, palette="exotic", height=3.0, style="smooth", decor="fruit"),
    dict(seed=42, palette="autumn", height=1.6, style="faceted", decor=None),
]


@pytest.mark.parametrize("kw", _TREE_CASES)
def test_generate_tree_is_deterministic(kw):
    a = propgen.generate_tree(**kw)
    b = propgen.generate_tree(**kw)
    for arr_a, arr_b in zip(a, b):
        assert arr_a.shape == arr_b.shape
        assert arr_a.dtype == arr_b.dtype
        assert np.array_equal(arr_a, arr_b), (
            f"generate_tree({kw}) is not deterministic — the model cache and "
            "every placed prop's look depend on it")


def test_generate_palm_is_deterministic():
    a = propgen.generate_palm(11, "green", 2.8)
    b = propgen.generate_palm(11, "green", 2.8)
    for arr_a, arr_b in zip(a, b):
        assert np.array_equal(arr_a, arr_b)


def test_different_seeds_give_different_trees():
    v1, _, _ = propgen.generate_tree(1)
    v2, _, _ = propgen.generate_tree(2)
    assert not (v1.shape == v2.shape and np.array_equal(v1, v2))


# ---------------------------------------------------------------------------
# The frozen signature (critique F27 — P3's entity fields ARE these params)
# ---------------------------------------------------------------------------

def test_public_signatures_are_frozen():
    assert list(inspect.signature(propgen.generate_tree).parameters) == [
        "seed", "palette", "height", "style", "decor", "decor_density"]
    assert list(inspect.signature(propgen.generate_palm).parameters) == [
        "seed", "palette", "height"]


def test_generators_registry_matches_the_public_functions():
    assert propgen.GENERATORS["tree"] is propgen.generate_tree
    assert propgen.GENERATORS["palm"] is propgen.generate_palm


def test_palettes_carry_every_channel_each_generator_reads():
    needed = ("leaf_h", "leaf_s", "leaf_v", "bark_h", "bark_s", "bark_v",
              "flower_h", "flower_s", "flower_v",
              "fruit_h", "fruit_s", "fruit_v")
    for name, pal in propgen.PALETTES.items():
        missing = [k for k in needed if k not in pal]
        assert not missing, f"palette {name!r} is missing {missing}"
        for k in needed:
            lo, hi = pal[k]
            assert lo <= hi, f"palette {name!r} range {k} is inverted"


@pytest.mark.parametrize("palette", sorted(propgen.PALETTES))
@pytest.mark.parametrize("decor", [None, "flowers", "fruit"])
def test_every_palette_and_decor_combination_generates(palette, decor):
    v, n, c = propgen.generate_tree(3, palette, 2.2, "smooth", decor)
    assert len(v) > 0


# ---------------------------------------------------------------------------
# Array contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", _TREE_CASES)
def test_triangle_array_contract(kw):
    v, n, c = propgen.generate_tree(**kw)
    assert v.dtype == np.float32 and v.ndim == 2 and v.shape[1] == 3
    assert n.dtype == np.float32 and n.shape == v.shape
    assert c.dtype == np.uint8 and c.shape == (v.shape[0], 4)
    assert len(v) % 3 == 0, "non-indexed triangle soup must be a multiple of 3"
    assert np.isfinite(v).all() and np.isfinite(n).all()
    # Normals are unit length (the shader does N·L on them directly).
    lens = np.linalg.norm(n, axis=1)
    assert np.allclose(lens, 1.0, atol=2e-3)


def test_vertex_alpha_is_flutter_data_not_opacity():
    """Alpha spans the flutter range (rigid trunk .. full-flutter tuft). A
    consumer that treated it as opacity would render a see-through tree — the
    prop shader forces output alpha to 1.0 for exactly this reason."""
    _, _, c = propgen.generate_tree(1, "green", 2.2, "smooth", None)
    a = c[:, 3]
    assert a.min() == 0, "the trunk must carry flutter weight 0"
    assert a.max() >= 250, "leaf tufts must carry flutter weight ~1"


def test_trunk_base_sits_at_the_origin():
    """Placement puts the prop at (x_wpx, 0, y_wpx), so the mesh's foot must be
    at y = 0 or a tree would float / sink."""
    for seed in (1, 2, 3):
        v, _, _ = propgen.generate_tree(seed, height=2.2)
        # The base ring is perpendicular to a slightly tilted trunk, so it dips
        # a few mm; anything more would be a floating or buried tree.
        assert abs(float(v[:, 1].min())) < 0.01 * 2.2


def test_height_parameter_scales_the_mesh():
    small, _, _ = propgen.generate_tree(5, "green", 1.5)
    big, _, _ = propgen.generate_tree(5, "green", 3.0)
    h_small = float(small[:, 1].max())
    h_big = float(big[:, 1].max())
    assert h_big > 1.6 * h_small


# ---------------------------------------------------------------------------
# Decor is emitted, and it is CLUSTERED (P0 top-down finding)
# ---------------------------------------------------------------------------

def test_decor_actually_adds_geometry():
    bare, _, _ = propgen.generate_tree(9, "green", 2.2, "smooth", None)
    flow, _, _ = propgen.generate_tree(9, "green", 2.2, "smooth", "flowers")
    fruit, _, _ = propgen.generate_tree(9, "green", 2.2, "smooth", "fruit")
    assert len(flow) > len(bare)
    assert len(fruit) > len(bare)


def test_decor_density_zero_emits_no_decor():
    bare, _, _ = propgen.generate_tree(9, "green", 2.2, "smooth", None)
    off, _, _ = propgen.generate_tree(9, "green", 2.2, "smooth", "flowers",
                                      0.0)
    assert len(off) == len(bare)


def test_decor_is_clustered_not_scattered():
    """The P2 fix: decor must read as a few deliberate masses from straight
    above, not an even sprinkle. Measured structurally — take the decor
    vertices (the ones whose colour is far from every leaf/bark colour is
    fragile, so instead compare against the bare tree: the ADDED vertices are
    exactly the decor) and check they form tight groups, i.e. the mean
    nearest-neighbour distance is small relative to the canopy radius."""
    kw = dict(seed=9, palette="green", height=2.2, style="smooth")
    bare, _, _ = propgen.generate_tree(decor=None, **kw)
    with_flowers, _, _ = propgen.generate_tree(decor="flowers", **kw)

    # The generator emits blob geometry then its decor, blob by blob, so the
    # decor vertices are simply "everything the bare tree does not have" by
    # count per blob; a robust proxy is the set difference on position.
    bare_set = {tuple(p) for p in np.round(bare, 5)}
    decor_pts = np.array([p for p in with_flowers
                          if tuple(np.round(p, 5)) not in bare_set])
    assert len(decor_pts) >= 12, "no decor geometry found"

    # Element centres: every flower is 4 triangles = 12 verts; group by 12.
    n_full = (len(decor_pts) // 12) * 12
    centres = decor_pts[:n_full].reshape(-1, 12, 3).mean(axis=1)
    assert len(centres) >= 4

    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    canopy_radius = float(np.linalg.norm(
        with_flowers.max(axis=0) - with_flowers.min(axis=0))) / 2.0
    # Clustered: a typical blossom's nearest neighbour is much closer than an
    # even scatter over the canopy would put it.
    assert float(np.median(nn)) < 0.12 * canopy_radius


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------

def test_propgen_imports_nothing_from_simulation():
    src = (propgen.__file__)
    text = open(src, "r", encoding="utf-8").read()
    assert "import simulation" not in text
    assert "from simulation" not in text
    assert "import pyray" not in text, (
        "propgen is pure geometry — pyray belongs in static_props.py")
