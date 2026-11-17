"""agents/network.py -- network spec v1 as code: the soldier's network and
the value network (design 8; rung 2 design 2; plan section 4 items 3-5).

THE SOLDIER (:class:`SoldierNet`), part by part (design 8.1):

- **vector reader**: the soldier's own vector (``V`` numbers as served,
  each kind column read as its learned vector) through two layers;
- **entry readers**: ONE single layer per entry type (the list's ``type``
  column picks it), so every entry becomes a vector of the same width;
- **list attention**: pre-norm attention layers over the soldier's own
  token and every entry, the padding masked off as keys; no position is
  encoded, so an entry is read the same wherever it stands;
- **list summary**: the soldier's token asks the list one question (one
  attention read) and gets one vector back;
- **near / far readers**: four 3 x 3 convolutions each (the last three
  halve the window), then one layer; the windows come in as served
  (``float16`` planes and ``uint8`` planes, a code plane read as one
  channel per code) and are converted inside;
- **join**: the four results to one vector;
- **memory**: a GRU cell carried from decision to decision, its update
  gates started over a spread of time scales (:func:`_keep_start`), the
  heads reading it through a normalising layer; the spec can switch it
  off, and then there is no state at all;
- **heads**, in the action spec's order (design 2.3): the feet on their
  own; the aim by pointing (a question vector against each entry's own
  vector and the learned vectors of the fixed tokens); the trigger and the
  hand told the chosen aim (its vector); the hand by pointing at a learned
  vector per verb (design 2.5); the argument last, told the aim and the
  verb. Every head's last layer starts a hundred times smaller than
  PyTorch's default (:data:`FIRST_HEAD_SCALE`), so a fresh network gives
  every legal option nearly the same chance (note 2b, 6).

A normalising layer (LayerNorm) follows each reader and precedes each
attention layer; no dropout; nothing reads another row of the batch.

THE VALUE NETWORK (:class:`ValueNet`, design 8.4, 7.1): the same readers
with their own weights, over the full view: the soldier's own vector with
the match block appended, the full list instead of the soldier's, the
soldier's two windows read a second time with the watched-by-everyone
plane beside the near one; no memory; after the join one layer, then one
output per reward group in the environment's order.

The networks take the arrays AS SERVED (torch tensors of the served
dtypes; :func:`as_tensors` wraps numpy arrays without a copy) and refuse
any other number format, so acting and learning convert in one place. The
choosing of the five choices is :mod:`agents.choosing`'s, never here.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as Fn

from agents import _game  # noqa: F401
from agents.sizes import FULL_PLANE, Sizes
from agents.spec import NetSpec, SpecRecord
from simulation import obs_spec_v1 as O
from simulation.factions import ZOMBIE_TEAM

#: What the soldier's network reads of ``obs`` (``list_uid`` is the
#: decoder's, never a network input; the masks are the choosing module's).
SOLDIER_KEYS = ("vector", "list", "list_mask", "near_f", "near_u", "far_f",
                "far_u")
#: What the value network reads of ``obs`` and of ``info["full_view"]``.
VALUE_OBS_KEYS = ("vector", "near_f", "near_u", "far_f", "far_u")
VALUE_FULL_KEYS = ("full_list", "full_mask", "full_match", FULL_PLANE)
#: The heads' last layers start this share of PyTorch's default (a
#: near-even first policy; rung 2 note 2b, section 6).
FIRST_HEAD_SCALE = 0.01
#: The longest time scale the memory STARTS with, in decisions (see
#: :func:`_keep_start`). A start, not a limit: learning moves every gate.
MEMORY_LONGEST = 32


def _keep_start(cell: nn.GRUCell) -> None:
    """Start the memory cell's update gates over a spread of time scales
    (chrono initialisation: Tallec and Ollivier, "Can recurrent neural
    networks warp time?", ICLR 2018, section 2). PyTorch's default starts
    every unit at an even keep / replace, so a mark on the memory halves
    every decision and is gone before a reward can teach the cell to keep
    it (the finding of rung 2 patch 2d.1). Here unit ``i`` starts keeping
    ``T_i / (T_i + 1)`` of itself per decision, the ``T_i`` spread evenly in
    the logarithm from 1 to :data:`MEMORY_LONGEST` decisions: some units
    write fast and forget fast, some write slowly and hold. No random
    number is drawn (a network without memory, and every other weight of
    one with it, starts exactly as before)."""
    core = cell.hidden_size
    scales = torch.logspace(0.0, math.log10(MEMORY_LONGEST), core)
    with torch.no_grad():
        # GRUCell's biases are laid out (reset, update, new); the update
        # gate multiplies the OLD memory, so a positive bias keeps.
        cell.bias_ih[core:2 * core].zero_()
        cell.bias_hh[core:2 * core].copy_(torch.log(scales))


def served_formats(sizes: Sizes) -> dict:
    """Per key the served dtype and the trailing shape a network checks,
    asked from the game's own shape tables (``obs_spec_v1.shapes`` /
    ``full_shapes``); a list's entry count is left out (any ``E``)."""
    out = {}
    for k, (shape, dt) in O.shapes(1, 1, sizes.n_slots).items():
        out[k] = (np.dtype(dt), tuple(shape[2:]))
    for k, (shape, dt) in O.full_shapes(1, 1, sizes.n_teams).items():
        out[k] = (np.dtype(dt), tuple(shape[2:]))
    out["list"] = (out["list"][0], (None,) + out["list"][1][1:])
    out["list_mask"] = (out["list_mask"][0], (None,))
    return out


_TORCH = {np.dtype(np.float32): torch.float32,
          np.dtype(np.float16): torch.float16,
          np.dtype(np.uint8): torch.uint8,
          np.dtype(np.int32): torch.int32}


def as_tensors(arrays: dict, device=None) -> dict:
    """numpy arrays -> torch tensors of the SAME dtype (no conversion; a
    view where the device is the processor)."""
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


def _check(x, key, formats, lead: int):
    dt, tail = formats[key]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{key}: a torch tensor is wanted, got {type(x)}")
    if x.dtype != _TORCH[dt]:
        raise TypeError(f"{key}: served as {dt}, got {x.dtype} (the network "
                        f"converts inside; feed it as served)")
    got = tuple(x.shape[lead:])
    if len(got) != len(tail) or any(t is not None and t != g
                                    for t, g in zip(tail, got)):
        raise ValueError(f"{key}: trailing shape {got}, the spec serves "
                         f"{tail} (None = any)")


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
class KindTable(nn.Module):
    """Reads a table of columns as served (``float32``): the number columns
    pass through, each kind column becomes its learned vector (design 8.3,
    "a learned vector per kind"). Row 0 of a kind's embedding is "none"
    (-1); a value past its rows fails loudly in the embedding."""

    def __init__(self, n_columns: int, kinds: tuple, kind_dim: int):
        super().__init__()
        kind_pos = [k.pos for k in kinds]
        num = [i for i in range(n_columns) if i not in set(kind_pos)]
        self.register_buffer("num_idx", torch.tensor(num, dtype=torch.long),
                             persistent=False)
        self.kind_pos = tuple(kind_pos)
        self.zombie_rows = tuple(k.zombie_row for k in kinds)
        self.embeds = nn.ModuleList(nn.Embedding(k.rows, kind_dim)
                                    for k in kinds)
        self.out_dim = len(num) + len(kinds) * kind_dim

    def forward(self, x):
        parts = [x.index_select(-1, self.num_idx)]
        for pos, zrow, emb in zip(self.kind_pos, self.zombie_rows,
                                  self.embeds):
            v = x[..., pos].long()
            idx = v + 1
            if zrow >= 0:
                idx = torch.where(v == ZOMBIE_TEAM, torch.full_like(idx, zrow),
                                  idx)
            parts.append(emb(idx))
        return torch.cat(parts, dim=-1)


class Attention(nn.Module):
    """Multi-head scaled dot-product attention, keys masked by ``key_mask``
    (``(N, K)`` bool, True = a key that may be read). Written out so that
    masking is exactly ``-inf`` before the softmax and no fast path differs
    between shapes."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError(f"attention width {width} over {heads} heads")
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)

    def forward(self, xq, xkv, key_mask):
        n, lq, w = xq.shape
        lk = xkv.shape[1]
        h, dh = self.heads, w // self.heads
        q = self.q(xq).view(n, lq, h, dh).transpose(1, 2)
        k, v = self.kv(xkv).view(n, lk, 2, h, dh).unbind(dim=2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        s = s.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        a = torch.softmax(s, dim=-1)
        y = (a @ v).transpose(1, 2).reshape(n, lq, w)
        return self.out(y)


class AttentionLayer(nn.Module):
    """Pre-norm self-attention and inner layer, each with its residual."""

    def __init__(self, width: int, heads: int, ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.ReLU(),
                                nn.Linear(ff, width))

    def forward(self, x, mask):
        y = self.norm1(x)
        x = x + self.attn(y, y, mask)
        return x + self.ff(self.norm2(x))


class EntryReaders(nn.Module):
    """One single layer per entry type (design 8.1), picked by the entry's
    ``type`` column, then the normalising layer."""

    def __init__(self, in_dim: int, n_types: int, width: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(in_dim, width)
                                    for _ in range(n_types))
        self.norm = nn.LayerNorm(width)

    def forward(self, z, etype):
        out = None
        for t, layer in enumerate(self.layers):
            y = layer(z) * (etype == t).unsqueeze(-1).to(z.dtype)
            out = y if out is None else out + y
        return self.norm(Fn.relu(out))


class MapReader(nn.Module):
    """Four 3 x 3 convolutions (the last three halve the window), then one
    layer to a vector and the normalising layer (design 8.1)."""

    def __init__(self, channels: int, side: int, conv: tuple, width: int):
        super().__init__()
        layers, c_in, s = [], channels, side
        for i, c in enumerate(conv):
            stride = 1 if i == 0 else 2
            layers += [nn.Conv2d(c_in, c, 3, stride=stride, padding=1),
                       nn.ReLU()]
            c_in = c
            s = (s + 2 - 3) // stride + 1
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(c_in * s * s, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        y = self.conv(x).flatten(1)
        return self.norm(Fn.relu(self.fc(y)))


def window_planes(window, f, u, extra=None):
    """The served window (``f`` float16, ``u`` uint8, both ``(N, C, W,
    W)``) as one float32 stack: the float planes, then per byte plane one
    channel (its number) or one channel per code; then ``extra`` (a
    ``uint8`` ``(N, W, W)`` plane) if given. A COUNT plane is read as
    ``log(1 + count)``: the game squashes every ratio at its door (design
    8.3) but serves a count raw, and a horde in one block would otherwise
    stand tens of times above the flags beside it."""
    chans = [f.to(torch.float32)]
    for i, (name, codes) in enumerate(window.u_planes):
        p = u[:, i]
        if codes:
            chans += [(p == c).to(torch.float32).unsqueeze(1) for c in codes]
        elif name in window.count_planes:
            chans.append(torch.log1p(p.to(torch.float32)).unsqueeze(1))
        else:
            chans.append(p.to(torch.float32).unsqueeze(1))
    if extra is not None:
        chans.append(extra.to(torch.float32).unsqueeze(1))
    return torch.cat(chans, dim=1)


class Body(nn.Module):
    """The readers, the attention, the summary and the join: shared by
    both networks as CODE, never as weights (each network has its own)."""

    def __init__(self, net: NetSpec, sizes: Sizes, entry_width: int,
                 entry_kinds: tuple, vector_extra: int, near_extra: int):
        super().__init__()
        w = net.width
        self.sizes = sizes
        self.vector_kinds = KindTable(sizes.vector, sizes.vector_kinds,
                                      net.kind_dim)
        self.vector_reader = nn.Sequential(
            nn.Linear(self.vector_kinds.out_dim + vector_extra,
                      net.vec_hidden), nn.ReLU(),
            nn.Linear(net.vec_hidden, w), nn.ReLU(), nn.LayerNorm(w))
        self.entry_kinds = KindTable(entry_width, entry_kinds, net.kind_dim)
        self.entry_readers = EntryReaders(self.entry_kinds.out_dim,
                                          sizes.entry_types, w)
        self.attention = nn.ModuleList(
            AttentionLayer(w, net.attn_heads, net.attn_ff)
            for _ in range(net.attn_layers))
        self.attention_norm = nn.LayerNorm(w)
        self.summary_norm = nn.LayerNorm(w)
        self.summary = Attention(w, net.attn_heads)
        self.near_reader = MapReader(sizes.near.channels + near_extra,
                                     sizes.near.side, net.conv, w)
        self.far_reader = MapReader(sizes.far.channels, sizes.far.side,
                                    net.conv, w)
        self.join = nn.Sequential(nn.Linear(4 * w, net.join), nn.ReLU(),
                                  nn.LayerNorm(net.join))
        self._type_pos = next(f.pos for f in O.ENTRY_COLUMNS
                              if f.name == "type")

    def forward(self, vector, lst, lmask, near, far, vector_extra=None):
        """``vector`` (N, V), ``lst`` (N, E, Fw), ``lmask`` (N, E) uint8,
        ``near`` / ``far`` float32 stacks. Returns the joined vector
        ``(N, join)`` and the entries' own vectors ``(N, E, width)``."""
        v = self.vector_kinds(vector)
        if vector_extra is not None:
            v = torch.cat([v, vector_extra], dim=-1)
        me = self.vector_reader(v)                         # (N, w)
        z = self.entry_kinds(lst)
        ents = self.entry_readers(z, lst[..., self._type_pos].long())
        x = torch.cat([me.unsqueeze(1), ents], dim=1)     # (N, 1 + E, w)
        ones = torch.ones_like(lmask[:, :1], dtype=torch.bool)
        mask = torch.cat([ones, lmask != 0], dim=1)
        for layer in self.attention:
            x = layer(x, mask)
        x = self.attention_norm(x)
        q = self.summary_norm(x[:, :1])
        summ = self.summary(q, x, mask)[:, 0]
        joined = self.join(torch.cat(
            [me, summ, self.near_reader(near), self.far_reader(far)], dim=-1))
        return joined, x[:, 1:]


def _mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_out))


# ---------------------------------------------------------------------------
# The soldier
# ---------------------------------------------------------------------------
class HeadOutputs:
    """The network's raw outputs for ``N`` rows, read head by head in the
    action spec's order by :mod:`agents.choosing`: the feet and the aim
    on their own, the trigger and the hand given the chosen aim, the
    argument given the aim and the verb. Holds the core vector and the
    aim's keys (the fixed tokens' learned vectors, then each entry's)."""

    def __init__(self, net: "SoldierNet", core, aim_keys):
        self.net = net
        self.core = core                      # (N, core)
        self.aim_keys = aim_keys              # (N, 18 + E, width)

    @property
    def n_rows(self) -> int:
        return self.core.shape[0]

    def take(self, idx) -> "HeadOutputs":
        """The outputs of the rows ``idx`` only (the gradient flows through
        the selection): the learner replays whole stretches and evaluates
        the policy rows among them."""
        return HeadOutputs(self.net, self.core[idx], self.aim_keys[idx])

    def _aim_vec(self, aim):
        idx = aim.long().view(-1, 1, 1).expand(-1, 1, self.aim_keys.shape[-1])
        return self.aim_keys.gather(1, idx)[:, 0]

    def feet(self):
        return self.net.feet_head(self.core)

    def aim(self):
        q = self.net.aim_query(self.core)
        return (self.aim_keys @ q.unsqueeze(-1))[..., 0] / self.net.scale

    def trigger(self, aim):
        return self.net.trigger_head(
            torch.cat([self.core, self._aim_vec(aim)], dim=-1))

    def hand(self, aim):
        q = self.net.hand_query(
            torch.cat([self.core, self._aim_vec(aim)], dim=-1))
        return q @ self.net.verbs.t() / self.net.scale

    def arg(self, aim, hand):
        verb = self.net.verbs.index_select(0, hand.long())
        return self.net.arg_head(
            torch.cat([self.core, self._aim_vec(aim), verb], dim=-1))


class SoldierNet(nn.Module):
    """The soldier's network, spec ``net`` for the layout ``sizes``."""

    def __init__(self, net: NetSpec, sizes: Sizes):
        super().__init__()
        self.spec = net
        self.sizes = sizes
        self.record = SpecRecord.of(net, sizes)
        self.formats = served_formats(sizes)
        w, hh, core = net.width, net.head_hidden, net.core
        self.body = Body(net, sizes, sizes.entry, sizes.entry_kinds, 0, 0)
        self.memory = nn.GRUCell(net.join, core) if net.memory else None
        if self.memory is not None:
            _keep_start(self.memory)
            # The heads read the memory through a normalising layer (the
            # memory itself is carried as the cell made it). Without it the
            # heads see a vector about eight times smaller than the join
            # (the cell's output starts small, most units keeping), and a
            # cue the heads must learn to read is learnt about twice as
            # slowly as without memory (patch 2d.1b). No random number.
            self.memory_norm = nn.LayerNorm(core)
        n_feet, _, n_trig, n_hands, n_arg = sizes.heads
        self.feet_head = _mlp(core, hh, n_feet)
        self.aim_query = _mlp(core, hh, w)
        self.aim_entry_key = nn.Linear(w, w)
        self.aim_tokens = nn.Parameter(torch.randn(sizes.n_aim_tokens, w)
                                       / math.sqrt(w))
        self.trigger_head = _mlp(core + w, hh, n_trig)
        self.hand_query = _mlp(core + w, hh, w)
        self.verbs = nn.Parameter(torch.randn(n_hands, w) / math.sqrt(w))
        self.arg_head = _mlp(core + 2 * w, hh, n_arg)
        self.scale = math.sqrt(w)
        # A near-even first policy (rung 2 note 2b, 6): the last layer of
        # every head starts FIRST_HEAD_SCALE times PyTorch's default, so
        # every legal option starts near the same chance.
        with torch.no_grad():
            for mlp in (self.feet_head, self.aim_query, self.trigger_head,
                        self.hand_query, self.arg_head):
                last = mlp[-1]
                last.weight.mul_(FIRST_HEAD_SCALE)
                last.bias.mul_(FIRST_HEAD_SCALE)

    # -- state --------------------------------------------------------------
    def initial_state(self, n: int, device=None):
        """The memory at a match's first decision: zeros, or None when the
        spec has no memory."""
        if self.memory is None:
            return None
        return torch.zeros(n, self.memory.hidden_size, device=device)

    # -- reading --------------------------------------------------------------
    def _read(self, obs: dict, lead: int):
        bad = [k for k in O.FULL_KEYS if k in obs]
        if bad:
            raise ValueError(f"the soldier's network never reads the full "
                             f"view (got {bad})")
        for k in SOLDIER_KEYS:
            _check(obs[k], k, self.formats, lead)
        flat = {k: obs[k].reshape((-1,) + tuple(obs[k].shape[lead:]))
                for k in SOLDIER_KEYS}
        s = self.sizes
        near = window_planes(s.near, flat["near_f"], flat["near_u"])
        far = window_planes(s.far, flat["far_f"], flat["far_u"])
        return self.body(flat["vector"], flat["list"], flat["list_mask"],
                         near, far)

    def _outputs(self, core, ents) -> HeadOutputs:
        n = ents.shape[0]
        keys = torch.cat([self.aim_tokens.unsqueeze(0).expand(n, -1, -1),
                          self.aim_entry_key(ents)], dim=1)
        return HeadOutputs(self, core, keys)

    def _step_memory(self, joined, state, first):
        if self.memory is None:
            if state is not None:
                raise ValueError("this network has no memory: no state")
            return joined, None
        if state is None:
            state = self.initial_state(joined.shape[0], joined.device)
        if first is not None:
            state = state * (~first.bool()).unsqueeze(-1).to(state.dtype)
        h = self.memory(joined, state)
        return self.memory_norm(h), h

    def forward(self, obs: dict, state=None, first=None):
        """One decision for ``N`` rows: ``obs`` the served arrays ``(N,
        ...)``, ``state`` the memory ``(N, core)`` (None: zeros), ``first``
        ``(N,)`` True where a match starts (the memory is zeroed there).
        Returns ``(HeadOutputs, new state)``."""
        joined, ents = self._read(obs, 1)
        core, new = self._step_memory(joined, state, first)
        return self._outputs(core, ents), new

    def forward_sequence(self, obs: dict, state=None, first=None):
        """A stretch of ``L`` decisions for ``N`` rows: ``obs`` ``(L, N,
        ...)``, ``state`` the memory at its start, ``first`` ``(L, N)``.
        The memory is stepped exactly as :meth:`forward` steps it. Returns
        ``(HeadOutputs`` over the ``L * N`` rows in ``(L, N)`` order, the
        memory after the last step)."""
        L, n = obs["vector"].shape[:2]
        joined, ents = self._read(obs, 2)
        if self.memory is None:
            if state is not None:
                raise ValueError("this network has no memory: no state")
            return self._outputs(joined, ents), None
        joined = joined.view(L, n, -1)
        outs = []
        for t in range(L):
            h, state = self._step_memory(joined[t], state,
                                         None if first is None else first[t])
            outs.append(h)
        core = torch.stack(outs).reshape(L * n, -1)
        return self._outputs(core, ents), state


# ---------------------------------------------------------------------------
# The value network
# ---------------------------------------------------------------------------
class ValueNet(nn.Module):
    """The training-only value network over the full view (design 8.4):
    one output per reward group, in :attr:`groups` order (the env's)."""

    def __init__(self, net: NetSpec, sizes: Sizes):
        super().__init__()
        self.spec = net
        self.sizes = sizes
        self.record = SpecRecord.of(net, sizes, value=True)
        self.formats = served_formats(sizes)
        self.groups = tuple(sizes.groups)
        self.body = Body(net, sizes, sizes.full_entry, sizes.full_kinds,
                         sizes.n_match, 1)
        self.out = nn.Sequential(nn.Linear(net.join, net.value_hidden),
                                 nn.ReLU(),
                                 nn.Linear(net.value_hidden, len(self.groups)))

    def forward(self, obs: dict, full: dict):
        """``obs`` the soldier's served arrays and ``full`` the served
        ``info["full_view"]``, both ``(N, ...)``: returns ``(N, groups)``."""
        for k in VALUE_OBS_KEYS:
            _check(obs[k], k, self.formats, 1)
        for k in VALUE_FULL_KEYS:
            _check(full[k], k, self.formats, 1)
        s = self.sizes
        near = window_planes(s.near, obs["near_f"], obs["near_u"],
                             full[FULL_PLANE])
        far = window_planes(s.far, obs["far_f"], obs["far_u"])
        joined, _ = self.body(obs["vector"], full["full_list"],
                              full["full_mask"], near, far,
                              vector_extra=full["full_match"])
        return self.out(joined)


__all__ = ["SOLDIER_KEYS", "VALUE_OBS_KEYS", "VALUE_FULL_KEYS",
           "FIRST_HEAD_SCALE",
           "served_formats", "as_tensors", "KindTable", "Attention",
           "AttentionLayer", "EntryReaders", "MapReader", "window_planes",
           "Body", "HeadOutputs", "SoldierNet", "ValueNet"]
