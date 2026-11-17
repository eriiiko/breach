"""agents/spec.py -- network spec "v1" and the record of what a network
fits (rung 2 design 2; design 8; plan section 4 items 3, 5, 7).

A network is a NAMED spec. ``"v1"`` is design 8: the readers of the vector,
the list (one reader per entry type, two attention layers, one summary
question), the two windows (four 3 x 3 convolutions each), the join, a
running memory (a GRU) the spec can switch off, the five heads; and the
value network (the same readers with their own weights over the full view,
no memory, one layer, five outputs). :func:`v1` gives its full width,
:func:`reduced` the same code at a width the tests run on a processor in
seconds. Widths are the network's; every size of the game's comes from a
:class:`agents.sizes.Sizes`.

THE RECORD (:class:`SpecRecord`) names what a network was made for: the
network spec, the observation spec, the action spec, the full view's spec
(the value network's), the number of grenade kinds, the planes of both
windows and the kind columns' rows; the value network's also the marine
teams of its match block. Loading a network into anything else is refused
by :func:`check_fits`, naming each field that differs. Not the list's
length: network v1 runs at any team size. (Files on disk are patch 2d's.)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from agents.sizes import Sizes

NETWORK_SPEC = "v1"


@dataclass(frozen=True)
class NetSpec:
    """The widths of network spec ``name``. ``memory`` switches the GRU."""
    name: str = NETWORK_SPEC
    width: int = 128            # entry, attention, summary, reader outputs
    kind_dim: int = 8           # one learned vector per kind value
    vec_hidden: int = 256       # the vector reader's first layer
    attn_layers: int = 2
    attn_heads: int = 4
    attn_ff: int = 256          # the attention layers' inner layer
    conv: tuple = (32, 64, 64, 64)
    join: int = 256             # the join, the memory, the value layer
    memory: bool = True
    head_hidden: int = 128
    value_hidden: int = 256

    @property
    def core(self) -> int:
        return self.join


def v1(memory: bool = True) -> NetSpec:
    """Network spec v1 at full width (design 8.1)."""
    return NetSpec(memory=bool(memory))


def reduced(memory: bool = True) -> NetSpec:
    """The same code at a reduced width, for the tests (design 2)."""
    return replace(v1(memory), width=32, kind_dim=4, vec_hidden=64,
                   attn_heads=2, attn_ff=64, conv=(8, 16, 16, 16), join=64,
                   head_hidden=32, value_hidden=64)


class SpecMismatch(ValueError):
    """A network was offered to a layout it was not made for."""


@dataclass(frozen=True)
class SpecRecord:
    """What a network fits (see the module docstring)."""
    network: str
    observation: str
    action: str
    full_view: str
    n_kinds: int
    near_planes: tuple
    far_planes: tuple
    kinds: tuple                # ((table, column, rows), ...)
    n_teams: int = -1           # the value network's match block; -1: none

    @staticmethod
    def of(net: NetSpec, sizes: Sizes, value: bool = False) -> "SpecRecord":
        """The record of a network of spec ``net`` made for ``sizes`` (the
        soldier's, or with ``value`` the value network's)."""
        def planes(w):
            return tuple(w.f_planes) + tuple(n for n, _ in w.u_planes)
        kinds = tuple(("vector", k.name, k.rows) for k in sizes.vector_kinds)
        kinds += tuple(("entry", k.name, k.rows) for k in sizes.entry_kinds)
        if value:
            kinds += tuple(("full", k.name, k.rows) for k in sizes.full_kinds)
        return SpecRecord(
            network=net.name, observation=sizes.obs_spec,
            action=sizes.action_spec, full_view=sizes.full_view,
            n_kinds=sizes.n_kinds, near_planes=planes(sizes.near),
            far_planes=planes(sizes.far), kinds=kinds,
            n_teams=sizes.n_teams if value else -1)

    def as_dict(self) -> dict:
        return asdict(self)


def check_fits(have: SpecRecord, want: SpecRecord) -> None:
    """Refuse ``have`` (a network's record) for a layout whose record is
    ``want``, naming every field that differs."""
    a, b = have.as_dict(), want.as_dict()
    bad = [k for k in a if a[k] != b[k]]
    if bad:
        detail = "; ".join(f"{k}: network {a[k]!r}, layout {b[k]!r}"
                           for k in bad)
        raise SpecMismatch(f"network does not fit this layout ({detail})")


__all__ = ["NETWORK_SPEC", "NetSpec", "v1", "reduced", "SpecMismatch",
           "SpecRecord", "check_fits"]
