#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the 8 paper equations as PNGs via LaTeX (Computer Modern math font),
identical to the compiled manuscript. Eq (1)-(6): display, centered, numbered.
Eq (7),(8): inline formulas in the paper -> centered, unnumbered."""
import os, subprocess, shutil, sys, tempfile
from PIL import Image

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "figures", "journal"))
TMP = tempfile.mkdtemp(prefix="eqtmp_")

PRE = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb}
\usepackage[paperwidth=24cm,paperheight=12cm,margin=6mm]{geometry}
\pagestyle{empty}
\newcommand{\JC}{J_{C}}
\newcommand{\etaSLA}{\eta^{\mathrm{SLA}}}
\begin{document}
\vspace*{2mm}
\begin{center}
"""
POST = r"""
\end{center}
\end{document}
"""

# (name, body, numbered)
EQS = [
    ("eq1", r"B_{i,s,t} = F_{s,t} + \max(0,\, q_{i,s,t} - \bar{Q}_s)\cdot p_{s,t}", 1),
    ("eq2", r"\begin{aligned} h^{-}_{s,t} &= \sigma\!\big(\theta_{0,s}(m) + \theta_{F,s}\varphi_{s,t} + \theta_{p,s}\psi_{s,t} - \theta_{\eta,s}\eta_{s,t-1}\big), \\ N^{\mathrm{leave}}_{s,t} &\sim \mathrm{Binomial}\big(N_{s,t},\, h^{-}_{s,t}\big) \end{aligned}", 2),
    ("eq3", r"\begin{aligned} h^{+}_{s,t} &= \sigma\!\big(\xi_{0,s} - \xi_{F,s}\varphi_{s,t} - \xi_{p,s}\psi_{s,t}\big), \\ N^{\mathrm{join}}_{s,t} &\sim \mathrm{Poisson}\big(\Lambda_s\, h^{+}_{s,t}\big) \end{aligned}", 3),
    ("eq4", r"\eta_{s,t} = \mathrm{clip}\big(1 - \alpha_s\max(0,\,u_{s,t}-\rho^{*}_s) + \varepsilon_{s,t},\; 0,\, 1\big)", 4),
    ("eq5", r"c_t = \sum_s N_{s,t}\cdot\max\big(0,\, \etaSLA_s - \eta_{s,t}\big)", 5),
    ("eq6", r"\max_\pi\; \mathbb{E}\Big[\textstyle\sum_t \gamma^t r_t\Big]\quad \text{s.t.}\quad \JC(\pi)=\mathbb{E}\Big[\textstyle\sum_t c_t\Big]\le d", 6),
    ("eq7", r"L(\pi,\lambda)=\mathbb{E}\Big[\sum\gamma^t(r_t-\lambda c_t)\Big]+\lambda d", 0),
    ("eq8", r"\lambda^{*}(m)=\Delta R/\Delta(\JC/d)", 0),
]

def build(name, body, num):
    if num:
        math = "\\[\n" + body + f"\\qquad\\qquad(\\mathrm{{{num}}})\n\\]\n"
    else:
        math = "\\[\n" + body + "\n\\]\n"
    tex = PRE + math + POST
    base = os.path.join(TMP, name)
    open(base + ".tex", "w").write(tex)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "-output-directory", TMP, base + ".tex"],
                       capture_output=True, text=True)
    if not os.path.exists(base + ".pdf"):
        print(f"FAIL compile {name}:\n{r.stdout[-800:]}"); return None
    # PDF -> PNG (300 dpi)
    subprocess.run(["pdftoppm", "-png", "-r", "300", base + ".pdf", base],
                   capture_output=True)
    png = base + "-1.png"
    if not os.path.exists(png):
        print(f"FAIL rasterize {name}"); return None
    im = Image.open(png).convert("RGB")
    # find non-white bbox
    from PIL import ImageChops
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox is None:
        print(f"FAIL empty {name}"); return None
    l, t, rr, b = bbox
    pad = 14
    # tight crop all sides (equation + appended number for numbered ones)
    crop = im.crop((max(0, l - pad), max(0, t - pad),
                    min(im.size[0], rr + pad), min(im.size[1], b + pad)))
    outp = os.path.join(OUT, name + ".png")
    crop.save(outp)
    return outp, crop.size

if __name__ == "__main__":
    if shutil.which("pdflatex") is None:
        sys.exit("pdflatex not found")
    for name, body, num in EQS:
        res = build(name, body, num)
        if res:
            print(f"wrote {res[0]}  {res[1][0]}x{res[1][1]}px  {'numbered' if num else 'inline(no number)'}")
