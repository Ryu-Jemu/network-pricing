#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the 5 journal tables as clean booktabs-style PNGs for HWP insertion.
Data mirrors KTCCS_SUBMISSION_PACKAGE.md §5 (audited 78/78)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "journal")
OUT = os.path.abspath(OUT)

TABLES = [
    dict(
        name="table1",
        caption="Table 1. Episode J_C and remaining eMBB base for constant\n"
                "policies at m=1 (static mapping).",
        headers=["Policy (m=1)", "J_C", "eMBB N_E"],
        rows=[
            ["Price cap (min J_C)", "1,647", "1,670"],
            ["Shed eMBB", "8,323", "4,278"],
            ["Reference price", "34,365", "5,039"],
            ["Grow eMBB (max J_C)", "37,578", "5,083"],
        ],
        align=["left", "right", "right"],
    ),
    dict(
        name="table2",
        caption="Table 2. Paired evaluation of static/heuristic policies at m=1.\n"
                "J_C above d_1=2,625 violates the SLA; the unconstrained\n"
                "top-revenue policies (BO, corner) all exceed the threshold.",
        headers=["Policy (m=1)", "Net reward R", "SLA cost J_C"],
        rows=[
            ["BO static oracle", "9,076", "4,759"],
            ["Corner [1,0,0,1]", "8,028", "23,578"],
            ["Max-Price (price cap)", "7,445", "1,650"],
            ["Grid static oracle", "6,337", "18,226"],
            ["Load threshold", "6,337", "18,226"],
            ["Peak/off-peak", "4,029", "29,513"],
            ["Reference price", "3,317", "33,843"],
            ["Zero price", "−44", "38,648"],
        ],
        align=["left", "right", "right"],
    ),
    dict(
        name="table3",
        caption="Table 3. Constraint satisfaction and price of safety for\n"
                "PPO-Lagrangian (vs. unconstrained PPO).",
        headers=["m", "R (mean)", "Satisf.", "Price of safety"],
        rows=[
            ["1", "7,482 [7,436, 7,563]", "0.99", "+0.7% (n.s., p=0.274)"],
            ["3", "6,924 [6,529, 7,233]", "0.93", "−10.9% (p=0.005)"],
            ["5", "7,197 [6,977, 7,354]", "0.54", "−3.7% (p=0.045)"],
            ["10", "1,754 [1,659, 1,877]", "1.00", "−74.4% (p<0.001)"],
        ],
        align=["center", "center", "center", "left"],
    ),
    dict(
        name="table4",
        caption="Table 4. PPO-Lagrangian advantage over the constraint-satisfying\n"
                "static oracle (one-sample t-test of seed means against the oracle constant).",
        headers=["m", "Static oracle R", "Advantage"],
        rows=[
            ["1", "7,445 (1/630 feasible)", "+0.5% (n.s., p=0.40)"],
            ["3", "6,796", "+1.9% (n.s., p=0.53)"],
            ["5", "6,620", "+8.7% (p=0.002)"],
            ["10", "5,006", "−65.0% (p<0.001)"],
        ],
        align=["center", "center", "left"],
    ),
    dict(
        name="table5",
        caption="Table 5. Algorithm comparison (endogenous env, unconstrained).",
        headers=["Algorithm", "m", "R (mean±std)", "J_C (mean)"],
        rows=[
            ["SAC", "1", "7,922 ± 326", "2,455"],
            ["SAC", "3", "6,770 ± 451", "1,791"],
            ["TD3", "1", "7,368 ± 438", "2,498"],
            ["TD3", "3", "5,868 ± 2,207", "7,381"],
        ],
        align=["center", "center", "center", "center"],
    ),
]

FS = 11.5           # cell font size
CAP_FS = 12.0       # caption font size
ROW_H = 0.42        # inch per row
CHAR_W = 0.092      # inch per character (column width estimate)
PAD = 0.35


def render(t):
    headers, rows, align = t["headers"], t["rows"], t["align"]
    ncol = len(headers)
    # column widths from longest cell
    colw = []
    for c in range(ncol):
        longest = max([len(headers[c])] + [len(r[c]) for r in rows])
        colw.append(max(longest * CHAR_W + 0.25, 0.7))
    total_w = sum(colw) + 2 * PAD
    cap_lines = t["caption"].count("\n") + 1
    total_h = (len(rows) + 1) * ROW_H + cap_lines * 0.30 + 0.5

    fig = plt.figure(figsize=(total_w, total_h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    # caption (top, left)
    ax.text(PAD / total_w, 1 - 0.06, t["caption"], transform=ax.transAxes,
            fontsize=CAP_FS, va="top", ha="left", family="DejaVu Sans")

    # table area below caption
    tbl_top = 1 - (cap_lines * 0.30 + 0.30) / total_h
    tbl_bottom = 0.10 / total_h
    tbl = ax.table(
        cellText=rows, colLabels=headers, cellLoc="center",
        colWidths=[w / sum(colw) for w in colw],
        bbox=[PAD / total_w, tbl_bottom,
              (total_w - 2 * PAD) / total_w, tbl_top - tbl_bottom],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FS)

    nrow = len(rows)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("none")
        cell.set_linewidth(0)
        cell.PAD = 0.04
        txt = cell.get_text()
        txt.set_ha({"left": "left", "right": "right", "center": "center"}[align[c]])
        if align[c] == "left":
            cell._loc = "left"
        elif align[c] == "right":
            cell._loc = "right"
        cell.set_height(1.0 / (nrow + 1))
        if r == 0:  # header row (colLabels are row 0)
            txt.set_weight("bold")
            cell.visible_edges = "TB"
            cell.set_edgecolor("black"); cell.set_linewidth(1.3)
        elif r == nrow:  # last data row -> bottom rule
            cell.visible_edges = "B"
            cell.set_edgecolor("black"); cell.set_linewidth(1.3)
    out = os.path.join(OUT, t["name"] + ".png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for t in TABLES:
        p = render(t)
        print("wrote", p)
