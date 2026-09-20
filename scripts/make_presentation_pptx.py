"""Generate the Traffic LSTM presentation deck (.pptx) from verified artifacts.

Every number on these slides comes from models/*_artifacts.json and
models/benchmark_*.json — the stored runs, not prose.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ---- palette (matches the web deck) ----
BG = RGBColor(0x0B, 0x11, 0x17)
BG_CARD = RGBColor(0x10, 0x1A, 0x22)
PAPER = RGBColor(0xF4, 0xEF, 0xE6)
STEEL = RGBColor(0x8F, 0xA3, 0xB8)
TEAL = RGBColor(0x2E, 0xC4, 0xB6)
AMBER = RGBColor(0xF0, 0xA2, 0x02)
CORAL = RGBColor(0xFF, 0x7A, 0x59)
RED = RGBColor(0xFF, 0x5A, 0x5F)

FONT_DISPLAY = "Georgia"
FONT_UI = "Calibri"
FONT_MONO = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

SW, SH = prs.slide_width, prs.slide_height


def add_slide():
    slide = prs.slides.add_slide(BLANK)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    return slide


def box(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.text_frame.word_wrap = True
    return tb


def text(slide, x, y, w, h, content, size=18, color=PAPER, bold=False,
         font=FONT_UI, align=PP_ALIGN.LEFT, line_spacing=1.0):
    h = max(0.3, min(h, 7.45 - y))
    tb = box(slide, x, y, w, h)
    tf = tb.text_frame
    lines = content.split("\n") if isinstance(content, str) else content
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.line_spacing = line_spacing
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
            r.font.name = font
    return tb


def rect(slide, x, y, w, h, fill, line_color=None):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.08
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line_color:
        sh.line.color.rgb = line_color
        sh.line.width = Pt(1)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def kicker(slide, txt, x=0.9, y=0.55):
    text(slide, x, y, 8, 0.4, txt.upper(), size=13, color=AMBER, bold=True, font=FONT_MONO)


def title(slide, txt, x=0.9, y=0.95, size=40, w=11.5):
    text(slide, x, y, w, 1.6, txt, size=size, color=PAPER, bold=True, font=FONT_DISPLAY)


def accent_bar(slide, x=0.9, y=0.62, w=0.09, h=0.55):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x - 0.18), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = TEAL
    sh.line.fill.background()
    sh.shadow.inherit = False


def bullets(slide, items, x=0.9, y=2.1, w=11.5, size=17, gap=6, color=PAPER):
    max_h = 7.4 - y
    tb = box(slide, x, y, w, max(1.0, min(5.0, max_h)))
    tf = tb.text_frame
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            head, body = item
            r = p.add_run()
            r.text = "▸ " + head + " — "
            r.font.color.rgb = TEAL
            r.font.bold = True
            r.font.size = Pt(size)
            r.font.name = FONT_UI
            r2 = p.add_run()
            r2.text = body
            r2.font.color.rgb = color
            r2.font.size = Pt(size)
            r2.font.name = FONT_UI
        else:
            p.text = "▸ " + item
            for r in p.runs:
                r.font.size = Pt(size)
                r.font.color.rgb = color
                r.font.name = FONT_UI
        p.space_after = Pt(gap)
        p.line_spacing = 1.12
    return tb


def metric_card(slide, x, y, w, h, value, label, note="", accent=AMBER):
    rect(slide, x, y, w, h, BG_CARD)
    text(slide, x + 0.25, y + 0.18, w - 0.5, 0.9, value, size=34, color=accent, bold=True, font=FONT_DISPLAY)
    text(slide, x + 0.25, y + 1.0, w - 0.5, 0.5, label, size=14, color=PAPER, bold=True)
    if note:
        text(slide, x + 0.25, y + 1.42, w - 0.5, 0.6, note, size=11, color=STEEL)


def footer(slide, txt, page):
    text(slide, 0.9, 7.0, 10, 0.4, txt, size=10, color=STEEL)
    text(slide, 12.3, 7.0, 0.7, 0.4, str(page), size=10, color=STEEL, align=PP_ALIGN.RIGHT, font=FONT_MONO)


def table(slide, x, y, w, rows, col_widths, header_fill=BG_CARD, size=13):
    from pptx.util import Inches as I
    shape = slide.shapes.add_table(len(rows), len(rows[0]), I(x), I(y), I(w), I(0.4 * len(rows)))
    tbl = shape.table
    for ci, cw in enumerate(col_widths):
        tbl.columns[ci].width = I(cw)
    for ri, row in enumerate(rows):
        for ci, cell_text in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = str(cell_text)
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if ri == 0 else BG
            cell.margin_top = cell.margin_bottom = Pt(4)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.name = FONT_UI
                    r.font.bold = ri == 0
                    r.font.color.rgb = PAPER if ri == 0 else (STEEL if ri > 0 else PAPER)
                    if ri > 0:
                        r.font.color.rgb = PAPER
    return tbl


PAGE = 0


def pg():
    global PAGE
    PAGE += 1
    return PAGE


# ================= 1 · TITLE =================
s = add_slide()
rect(s, 0, 0, 13.333, 0.12, TEAL)
text(s, 0.9, 1.6, 11, 0.5, "TRAFFIC OBSERVATORY · MACHINE LEARNING PROJECT", size=14, color=AMBER, bold=True, font=FONT_MONO)
text(s, 0.9, 2.2, 11.5, 2.4, "Forecast the next hour of traffic.\nSee every neuron fire.", size=48, color=PAPER, bold=True, font=FONT_DISPLAY)
text(s, 0.9, 4.6, 10.5, 1.2,
     "A stacked LSTM trained on 40,575 hours of motorway traffic — evaluated honestly against\n"
     "naive rules and gradient boosting, then opened gate by gate so the mechanism is visible.",
     size=17, color=STEEL, line_spacing=1.3)
text(s, 0.9, 6.3, 10, 0.5, "Raphael Girard · 2026", size=14, color=STEEL)
text(s, 10.2, 6.3, 2.4, 0.5, "github.com/GirardRaphael/lstm", size=11, color=TEAL, font=FONT_MONO)
pg()

# ================= 2 · THE PROBLEM =================
s = add_slide()
accent_bar(s)
kicker(s, "01 · The problem")
title(s, "Traffic is a time series with memory.")
bullets(s, [
    ("The task", "predict how many vehicles pass in the next hour, from the last 24 hours."),
    ("Why it matters", "signal timing, staffing, and incident response all consume a forecast — a wrong one costs real minutes."),
    ("The constraint", "in production you only ever have the past. Any evaluation that leaks the future is fiction."),
    ("The bar to clear", "the model must beat “last hour” and “same hour yesterday” — otherwise the honest answer is: don't use a neural network."),
], y=2.2, size=17, gap=14)
footer(s, "Supervised regression · target = vehicles per hour", pg())

# ================= 3 · WHY LSTM =================
s = add_slide()
accent_bar(s)
kicker(s, "02 · Why LSTM is the right algorithm here")
title(s, "Traffic has memory. LSTM is built for memory.")
bullets(s, [
    ("Ordered dependence", "rush hour builds over hours — 8h ago still matters. A feed-forward net sees a flat vector; an LSTM carries a cell state through the sequence and decides what to keep."),
    ("Selective memory is inspectable", "forget / input / output gates are explicit multiplicative valves. We can read them — and this project does, neuron by neuron."),
    ("Long-range without explosion", "the cell state's additive update (c = f·c + i·g) avoids the vanishing gradient that kills plain RNNs on 24-step sequences."),
    ("Multivariate & multi-horizon ready", "weather, calendar, and horizons 1/3/6 drop in as extra channels — no feature re-engineering."),
    ("Proven on this data", "beats both naive baselines by ~61% and wins outright on the second dataset (bike sharing)."),
], y=2.15, size=15.5, gap=11)
footer(s, "Hochreiter & Schmidhuber, 1997 — applied to hourly traffic counts", pg())

# ================= 4 · WHY NOT THE ALTERNATIVES =================
s = add_slide()
accent_bar(s)
kicker(s, "03 · The honest comparison")
title(s, "What else could we have used?")
table(s, 0.9, 2.0, 11.5, [
    ["Approach", "Why not (or when)", "Verdict on this data"],
    ["Persistence / seasonal naive", "No learning at all", "MAE 585.6 / 594.3 — floor to beat"],
    ["ARIMA-style linear models", "Linear, one series, no exogenous memory", "Cannot hold 24h of context"],
    ["XGBoost on lag features", "Strong tabular baseline — but windows are flattened; temporal order is implicit", "MAE 154.3 — wins on motorway"],
    ["Stacked LSTM", "Native sequence memory, inspectable gates", "MAE 201.4 — wins on bike sharing"],
], [3.2, 5.0, 3.3], size=12.5)
text(s, 0.9, 5.4, 11.5, 1.2,
     "Engineering stance: XGBoost is the champion baseline; the LSTM is the challenger that explains itself.\n"
     "Where temporal structure dominates (bike rentals), the LSTM wins. The tool choice is empirical — measured, not argued.",
     size=14, color=STEEL, line_spacing=1.3)
footer(s, "Both families evaluated on identical chronological tensors", pg())

# ================= 5 · DATASET =================
s = add_slide()
accent_bar(s)
kicker(s, "04 · The data")
title(s, "40,575 real hours of motorway traffic.")
metric_card(s, 0.9, 2.1, 2.7, 2.0, "48,204", "raw hourly rows", "UCI Metro Interstate")
metric_card(s, 3.8, 2.1, 2.7, 2.0, "40,575", "unique hours", "duplicates averaged")
metric_card(s, 6.7, 2.1, 2.7, 2.0, "6 years", "2012 → 2018", "I-94 Minneapolis–St Paul")
metric_card(s, 9.6, 2.1, 2.7, 2.0, "2,588", "gaps disclosed", "incl. a 308-day outage")
bullets(s, [
    ("Cleaning contract", "sort chronologically, collapse duplicate timestamps by averaging, never interpolate missing periods."),
    ("The disclosed flaw", "28.7% of training windows span a time gap — measured, published, and guarded with --drop-gapped-windows."),
], y=4.5, size=15, gap=10)
footer(s, "UCI Machine Learning Repository · Metro Interstate Traffic Volume", pg())

# ================= 6 · ARCHITECTURE =================
s = add_slide()
accent_bar(s)
kicker(s, "05 · The network")
title(s, "24 hours in, one number out.")
arch = [("24 past hours", "input window"), ("LSTM 64", "first memory"), ("LSTM 32", "compressed memory"),
        ("Dense 16", "decision head"), ("1 forecast", "vehicles next hour")]
x = 0.9
for i, (name, sub) in enumerate(arch):
    rect(s, x, 2.3, 2.05, 1.15, BG_CARD, line_color=TEAL if i in (1, 2) else None)
    text(s, x, 2.5, 2.05, 0.5, name, size=16, color=PAPER, bold=True, align=PP_ALIGN.CENTER)
    text(s, x, 2.95, 2.05, 0.4, sub, size=11, color=STEEL, align=PP_ALIGN.CENTER)
    if i < len(arch) - 1:
        text(s, x + 2.05, 2.55, 0.35, 0.5, "→", size=20, color=TEAL, align=PP_ALIGN.CENTER)
    x += 2.4
bullets(s, [
    ("89,573 parameters", "64-unit layer reads the day; 32-unit layer compresses it; the dense head prices it."),
    ("Dropout 0.20 between layers", "regularisation chosen by ablation, not habit."),
    ("Linear output", "regression — the answer is a count, not a class."),
    ("Adam + early stopping", "validation loss decides when to quit; the calendar run stopped at epoch 36 of 100."),
], y=3.9, size=15, gap=9)
footer(s, "Keras / TensorFlow 2.21 · CPU", pg())

# ================= 7 · CAUSAL PIPELINE =================
s = add_slide()
accent_bar(s)
kicker(s, "06 · Pipeline honesty (v2)")
title(s, "Split first. Fit nothing on the future.")
bullets(s, [
    ("Chronological partitions", "raw timestamps split into fit / validation / test before any transform is fitted — the scaler never sees validation or test rows."),
    ("Complete windows or nothing", "a window with a missing hour is excluded and counted by reason — never imputed."),
    ("Boundary purge", "any training example whose target crosses into validation is dropped. History may be context; it may not be a label."),
    ("Timestamp-true baselines", "“same hour yesterday” is looked up by wall-clock timestamp, not row offset."),
    ("Versioned, hash-verified packages", "model + scalers + units + cadence + dataset SHA256 travel together; corrupted artifacts are rejected on load."),
], y=2.15, size=15.5, gap=12)
footer(s, "18/18 temporal tests · 18/18 pipeline tests — verified 2026-09-20", pg())

# ================= 8 · RESULTS — MOTORWAY =================
s = add_slide()
accent_bar(s)
kicker(s, "07 · Results — motorway (MAE, vehicles/hour, lower is better)")
title(s, "The measured league table.")
table(s, 0.9, 2.0, 11.5, [
    ["Model", "Inputs", "MAE", "vs best naive", "Train time"],
    ["XGBoost + calendar", "traffic + hour/weekday", "154.3", "−73.7%", "14 s"],
    ["LSTM + calendar", "traffic + hour/weekday", "201.4", "−65.6%", "688 s"],
    ["LSTM traffic only", "24 past hours", "228.8", "−60.9%", "1,072 s"],
    ["Naive: same hour yesterday", "—", "594.3", "—", "—"],
    ["Naive: last hour", "—", "585.6", "—", "—"],
], [3.4, 3.3, 1.6, 1.8, 1.4], size=13)
text(s, 0.9, 5.5, 11.5, 1.0,
     "Every model beats the naive rules by a wide margin — the neural network earns its place.\n"
     "XGBoost is stronger on this dataset; that is a measured fact, and it is part of the story.",
     size=14, color=STEEL, line_spacing=1.3)
footer(s, "Identical chronological split · identical eligible windows", pg())

# ================= 9 · RESULTS — BIKE =================
s = add_slide()
accent_bar(s)
kicker(s, "08 · The verdict flips")
title(s, "Second dataset, opposite winner.")
table(s, 0.9, 2.1, 11.5, [
    ["Bike-sharing (hourly rentals)", "LSTM", "XGBoost", "Winner"],
    ["Univariate", "41.3", "43.1", "LSTM by 4.2%"],
    ["Multivariate (+ weather, calendar)", "38.5", "38.6", "LSTM by 0.3%"],
], [4.6, 2.2, 2.2, 2.5], size=14)
bullets(s, [
    ("Same code, same evaluation", "the only thing that changed is the dataset."),
    ("The lesson", "one dataset cannot settle “LSTM vs boosting” — the question is empirical, and this project measures it instead of asserting it."),
], y=4.3, size=15.5, gap=12)
footer(s, "Capital Bikeshare hourly · identical pipeline", pg())

# ================= 10 · ABLATION =================
s = add_slide()
accent_bar(s)
kicker(s, "09 · The ablation that settled it")
title(s, "The calendar is the signal. The weather is noise that costs.")
table(s, 0.9, 2.0, 11.5, [
    ["Inputs", "Columns", "Dropout", "LSTM MAE"],
    ["Past traffic only", "1", "0.20", "228.8"],
    ["Past traffic only (control)", "1", "0.35", "229.8"],
    ["Traffic + calendar", "7", "0.20", "201.4  ✓ best LSTM"],
    ["Traffic + calendar + weather", "11", "0.20", "241.1  ✗ overfit"],
    ["Traffic + calendar + weather", "11", "0.35", "206.0"],
], [4.4, 1.6, 1.6, 3.9], size=13)
bullets(s, [
    ("The control proves it", "same network, stronger dropout, no new features: 229.8 — nothing moves. It was never about regularisation."),
    ("Why weather hurts", "rain_1h and snow_1h are zero nearly every hour — parameters to overfit, no information to use. XGBoost ignores the column (154.3 with or without); the LSTM overfits it."),
], y=4.9, size=14, gap=10)
footer(s, "Validation loss bottomed at epoch 7 of 12 on the weather run — the signature of overfitting", pg())

# ================= 11 · INTERPRETABILITY =================
s = add_slide()
accent_bar(s)
kicker(s, "10 · Opening the black box")
title(s, "Every gate, every neuron, replayed by hand.")
bullets(s, [
    ("NumPy replay of the LSTM cell", "z = x·W + h·U + b → i, f, g, o gates; c = f·c + i·g; h = o·tanh(c) — re-implemented from the trained weights."),
    ("Verified, not illustrative", "the replay is checked against Keras on every export: max absolute difference 1.17e-7 — floating-point noise."),
    ("96 neuron profiles", "each unit gets its firing trace, memory score (mean forget gate), and a role label."),
    ("Rush hour vs quiet night", "different subsets of units light up in different regimes — the network has specialised, and you can see it."),
], y=2.15, size=15.5, gap=12)
footer(s, "Obsidian vault: 115 notes · 27 canvases · 990 wikilinks, 0 broken", pg())

# ================= 12 · LIVE DEMO =================
s = add_slide()
accent_bar(s)
kicker(s, "11 · The live demonstration")
title(s, "Jam the intersection. Watch it think.")
bullets(s, [
    ("2D intersection, real rules", "three lanes per approach — left / straight / right; cars stop at red, queue behind each other, and wait for their arrow."),
    ("Pedestrian crosswalks", "press NS or EW — the button lights up and the next all-red becomes a walk phase."),
    ("Traffic jams on demand", "inject 15–25 vehicles into one approach and watch queues form."),
    ("The decision brain", "live LSTM gates (forget / input / output), a neural graph that lights up, and a step-by-step decision trace: sense queue → update memory → forecast → stretch green."),
    ("Adaptive green", "green time follows queue pressure — 4 s minimum up to 16 s under a jam."),
], y=2.15, size=15, gap=11)
footer(s, "Simulation only — no connection to real signals or city sensors", pg())

# ================= 13 · LIMITS =================
s = add_slide()
accent_bar(s)
kicker(s, "12 · What it cannot do")
title(s, "The honest limits.")
bullets(s, [
    ("It forecasts; it does not control", "the output is a number a controller could consume — this project never touches signal hardware."),
    ("First-time events are invisible", "a model trained on history cannot predict the unprecedented."),
    ("XGBoost is stronger on the motorway data", "the LSTM is the interpretable challenger, not the universal champion — and the vault says so in writing."),
    ("Gaps are real", "28.7% of training windows span breaks in the series; disclosed, measured, and guarded — not hidden."),
], y=2.15, size=15.5, gap=13)
footer(s, "A worse score is often the informative one", pg())

# ================= 14 · CLOSE =================
s = add_slide()
rect(s, 0, 7.38, 13.333, 0.12, TEAL)
text(s, 0.9, 1.5, 11, 0.5, "SUMMARY", size=14, color=AMBER, bold=True, font=FONT_MONO)
text(s, 0.9, 2.1, 11.5, 1.6, "A forecaster you can interrogate.", size=44, color=PAPER, bold=True, font=FONT_DISPLAY)
bullets(s, [
    "Stacked LSTM, 24 hours → next-hour vehicles, 61–66% better than naive rules.",
    "Causal v2 pipeline: split before fit, zero future leakage, hash-verified packages.",
    "The mechanism is the deliverable: gates and neurons replayed and verified to 1e-7.",
    "Live demo: an intersection that obeys real traffic rules while the network's thinking stays on screen.",
], y=3.6, size=16, gap=12)
text(s, 0.9, 6.2, 11, 0.5, "github.com/GirardRaphael/lstm", size=14, color=TEAL, font=FONT_MONO)
pg()

out = "/workspace/presentation/Traffic_LSTM_Presentation.pptx"
prs.save(out)
print("saved", out)
