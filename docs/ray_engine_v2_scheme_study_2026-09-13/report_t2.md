# T2 — the currency audit: is one gas heat count one solid heat count?

> **Written incrementally** (design v2 §5, T2 row; the arc's standing rule that a
> kill must cost no resume). Branch `12-t2-currency-audit`, cut from `fire-12`
> at `6eab0f8`.
>
> **This patch changes no behaviour.** It is a measurement and an argument; the
> number it recommends is applied at **T5**.
>
> **Depends on**: `thermal_model_v2_design_2026-09-19.md` §3.2, R13, §4 ledger
> item 4, §5 T2 row, §6 item 5; `report_p2b.md` §1–§4 (the solid-side pin);
> arc #54's gas-energy seam and closure identity (CLAUDE.md, three rows).

---

## 0. Status

- [ ] §1 What the currency is, on both sides, read off the code
- [ ] §2 D1 — what `c_v` must be (the arithmetic)
- [ ] §3 D1 — the engine check (the measurement, not the paper)
- [ ] §4 D2 — where `c_v` belongs: the representation or the seam
- [ ] §5 D2 — the #54 closure identity under the recommendation
- [ ] §6 D3 — the blast radius, by file and line
- [ ] §7 `n_floor_heat` under a rescaled `c_v`
- [ ] §8 Open questions for Erik
- [ ] §9 What did not hold

