"""
Generate attention-map figures for the w9s1 + lookahead-sink config.

Config: local_attn_size=10, sink_size=1, chunk_size=3, lookahead_distance=4.

Two figures:
  - attn_map_w9s1.png        : plain mask, frame-index × frame-index.
  - attn_map_w9s1_rope.png   : same queries, but each chunk's sink (f_0)
                               is plotted at its VIRTUAL future frame
                               (= last-real-frame + lookahead_distance).
                               Non-sink keys keep their actual frame index.

Rationale: non-sink RoPE positions are just "within-window natural order",
which for any chunk's window equals the absolute frame order anyway. Only
the sink gets a deliberately-shifted virtual position. Showing non-sink at
actual frame indices makes the sliding eviction and the sink's "future"
placement visually obvious.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path(__file__).parent

# ── Config ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 3
LOCAL_ATTN_SIZE = 10
SINK_SIZE = 1
LOOKAHEAD_DISTANCE = 4
N_FRAMES = 15
N_CHUNKS = N_FRAMES // CHUNK_SIZE

# Max column we need to render = last chunk's sink virtual frame.
# For chunk c, last real frame = (c+1)*CHUNK_SIZE - 1, sink virtual = + lookahead.
LAST_REAL_FRAME = N_FRAMES - 1
MAX_SINK_VIRTUAL = LAST_REAL_FRAME + LOOKAHEAD_DISTANCE   # 18
N_COLS = MAX_SINK_VIRTUAL + 1                              # 19

SINK_COLOR = "#d73027"
NONSINK_COLOR = "#1f4b8e"
EMPTY_COLOR = "#f5f5f5"
VACATED_COLOR = "#fde0df"   # light pink — sink's vacated natural slot (col 0)
SKIPPED_BLOCK_BG = "#fff0ee"  # even lighter pink — "next block skipped" region


# ── Window construction per chunk (matches network_causal kv-cache logic) ─
def chunk_window(k: int) -> list:
    last_frame = (k + 1) * CHUNK_SIZE - 1
    visible = set(range(SINK_SIZE))
    non_sink_needed = LOCAL_ATTN_SIZE - SINK_SIZE
    first_non_sink = max(SINK_SIZE, last_frame - non_sink_needed + 1)
    visible.update(range(first_non_sink, last_frame + 1))
    return sorted(visible)


# ── Figure 1: plain frame×frame mask (unchanged) ──────────────────────────
mask_fxf = np.zeros((N_FRAMES, N_FRAMES), dtype=int)
for q in range(N_FRAMES):
    for k in chunk_window(q // CHUNK_SIZE):
        if k < N_FRAMES:
            mask_fxf[q, k] = 2 if k < SINK_SIZE else 1

fig, ax = plt.subplots(figsize=(8.8, 8.8))
cmap = plt.matplotlib.colors.ListedColormap([EMPTY_COLOR, NONSINK_COLOR, SINK_COLOR])
ax.imshow(mask_fxf, cmap=cmap, vmin=0, vmax=2, aspect="equal")
ax.set_xticks(np.arange(-0.5, N_FRAMES, 1), minor=True)
ax.set_yticks(np.arange(-0.5, N_FRAMES, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.8)
ax.tick_params(which="minor", length=0)
for c in range(1, N_CHUNKS + 1):
    pos = c * CHUNK_SIZE - 0.5
    ax.axhline(pos, color="black", linewidth=1.6)
    ax.axvline(pos, color="black", linewidth=1.6)
ax.add_patch(mpatches.Rectangle(
    (-0.5, -0.5), SINK_SIZE, N_FRAMES,
    linewidth=2.5, edgecolor=SINK_COLOR, facecolor="none", zorder=5,
))
ax.set_xticks(np.arange(N_FRAMES))
ax.set_yticks(np.arange(N_FRAMES))
ax.set_xticklabels([f"{i}" for i in range(N_FRAMES)], fontsize=9)
ax.set_yticklabels([f"{i}" for i in range(N_FRAMES)], fontsize=9)
ax.set_xlabel("Key frame index", fontsize=11)
ax.set_ylabel("Query frame index", fontsize=11)
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
for c in range(N_CHUNKS):
    y_mid = c * CHUNK_SIZE + (CHUNK_SIZE - 1) / 2
    ax.text(-1.6, y_mid, f"C{c}", fontsize=11, fontweight="bold",
            ha="center", va="center")
ax.set_title(
    "Attention mask — w9s1 + block-causal + sliding window\n"
    "(local_attn_size=10, sink_size=1, chunk_size=3)",
    fontsize=13, pad=30,
)
ax.legend(
    handles=[
        mpatches.Patch(facecolor=NONSINK_COLOR, label="non-sink attends"),
        mpatches.Patch(facecolor=SINK_COLOR, label="sink attends"),
        mpatches.Patch(facecolor=EMPTY_COLOR, edgecolor="#999", label="masked"),
    ],
    loc="lower right", bbox_to_anchor=(1.0, -0.12), frameon=False, fontsize=10,
)
plt.tight_layout()
out1 = OUT_DIR / "attn_map_w9s1.png"
plt.savefig(out1, dpi=160, bbox_inches="tight")
print(f"saved {out1}")
plt.close(fig)


# ── Figure 2: absolute frame index, sink plotted at virtual future frame ──
#
# Cell state enum:
#   0 = masked / not in window
#   1 = non-sink attends (at actual frame index)
#   2 = sink attends (at virtual future frame)
#   3 = sink's vacated natural slot (col 0 for C1+)
mask_abs = np.zeros((N_FRAMES, N_COLS), dtype=int)
cell_label = [["" for _ in range(N_COLS)] for _ in range(N_FRAMES)]
# Track sink virtual column per chunk for annotation
sink_virtual_col = {}

for q in range(N_FRAMES):
    c = q // CHUNK_SIZE
    win = chunk_window(c)
    has_lookahead = (c >= 1)
    last_abs = max(win)
    f_w = len(win)

    # Compute RoPE position of each frame in this chunk's window.
    # Matches network_causal._apply_window_rope:
    #   - lookahead ON : sink → F_w-1+lookahead,  non-sink → 1..F_w-1 in window order
    #   - lookahead OFF: all frames → 0..F_w-1 in window order
    non_sink = [f for f in win if f >= SINK_SIZE]
    rope_pos = {}
    if has_lookahead:
        rope_pos[0] = f_w - 1 + LOOKAHEAD_DISTANCE   # sink
        for slot_i, f in enumerate(non_sink, start=1):
            rope_pos[f] = slot_i
    else:
        for slot_i, f in enumerate(win):
            rope_pos[f] = slot_i

    for frame in win:
        if frame < SINK_SIZE:
            col = (last_abs + LOOKAHEAD_DISTANCE) if has_lookahead else frame
            sink_virtual_col[c] = col
            mask_abs[q, col] = 2
            cell_label[q][col] = f"{rope_pos[frame]}"
        else:
            mask_abs[q, frame] = 1
            cell_label[q][frame] = f"{rope_pos[frame]}"

    # Mark sink's vacated natural slot (col 0) for chunks with lookahead
    if has_lookahead and mask_abs[q, 0] == 0:
        mask_abs[q, 0] = 3


fig, ax = plt.subplots(figsize=(16, 9))
cmap4 = plt.matplotlib.colors.ListedColormap(
    [EMPTY_COLOR, NONSINK_COLOR, SINK_COLOR, VACATED_COLOR]
)

# Paint the "skipped block" background for each chunk (cols between last
# non-sink frame and sink virtual col) to make the gap visible
for c in range(N_CHUNKS):
    if c == 0:
        continue  # C0 has no lookahead
    last_abs = max(chunk_window(c))
    sink_col = sink_virtual_col[c]
    # Gap cols: (last_abs+1) .. (sink_col - 1)
    gap_start = last_abs + 0.5
    gap_end = sink_col - 0.5
    y0 = c * CHUNK_SIZE - 0.5
    y1 = (c + 1) * CHUNK_SIZE - 0.5
    ax.add_patch(mpatches.Rectangle(
        (gap_start, y0), gap_end - gap_start, y1 - y0,
        facecolor=SKIPPED_BLOCK_BG, edgecolor="none", zorder=0,
    ))

ax.imshow(mask_abs, cmap=cmap4, vmin=0, vmax=3, aspect="equal", zorder=1)

# Minor grid
ax.set_xticks(np.arange(-0.5, N_COLS, 1), minor=True)
ax.set_yticks(np.arange(-0.5, N_FRAMES, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.8)
ax.tick_params(which="minor", length=0)

# Query chunk boundaries (horizontal)
for c in range(1, N_CHUNKS + 1):
    ax.axhline(c * CHUNK_SIZE - 0.5, color="black", linewidth=1.6, zorder=3)

# Block boundaries (vertical) every CHUNK_SIZE frames — dotted
for b in range(CHUNK_SIZE, N_COLS, CHUNK_SIZE):
    ax.axvline(b - 0.5, color="#666666", linewidth=0.8, linestyle=":", zorder=2)

# Real/virtual boundary: dashed after frame N_FRAMES-1
ax.axvline(N_FRAMES - 0.5, color="#555555", linewidth=1.4,
           linestyle="--", alpha=0.8, zorder=2)
ax.text(N_FRAMES - 0.5, -1.4,
        "real frames  |  virtual (not generated)",
        fontsize=9, ha="center", va="bottom", color="#333", style="italic")

# Cell labels = RoPE position applied to each key during that chunk's attention
for q in range(N_FRAMES):
    for p in range(N_COLS):
        lbl = cell_label[q][p]
        if not lbl:
            continue
        is_sink = (mask_abs[q, p] == 2)
        ax.text(p, q, lbl, fontsize=8.5,
                color="white", fontweight="bold",
                ha="center", va="center", zorder=4)

# Sink cell outline (thick black)
for q in range(N_FRAMES):
    for p in range(N_COLS):
        if mask_abs[q, p] == 2:
            ax.add_patch(mpatches.Rectangle(
                (p - 0.5, q - 0.5), 1, 1,
                linewidth=2.2, edgecolor="black", facecolor="none", zorder=5,
            ))

# Axis ticks & labels
ax.set_xticks(np.arange(N_COLS))
ax.set_yticks(np.arange(N_FRAMES))
ax.set_xticklabels([f"{i}" for i in range(N_COLS)], fontsize=8.5)
ax.set_yticklabels([f"{i}" for i in range(N_FRAMES)], fontsize=9)
ax.set_xlabel("Key frame index (real + virtual future positions for sink)", fontsize=11)
ax.set_ylabel("Query frame index", fontsize=11)
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()

# Per-chunk row labels
for c in range(N_CHUNKS):
    y_mid = c * CHUNK_SIZE + (CHUNK_SIZE - 1) / 2
    f_w = len(chunk_window(c))
    if c == 0:
        note = f"F_w={f_w}\nno lookahead"
    else:
        sink_col = sink_virtual_col[c]
        last_abs = max(chunk_window(c))
        note = f"F_w={f_w}\nsink@f{sink_col}\n(={last_abs}+{LOOKAHEAD_DISTANCE})"
    ax.text(-2.2, y_mid, f"C{c}\n{note}",
            fontsize=8.5, fontweight="bold", ha="center", va="center")

# Annotate the skipped-block gap for the steady-state C3 row visually
# (cols 12, 13, 14 for C3, which is queries 9-11)
# Already shaded by the background rectangle above. Add a label pointer:
ax.annotate(
    "← 1 block skipped (virtual f12-f14)\n     sink lands at f15",
    xy=(14.5, 10),
    xytext=(15.5, 6.5),
    fontsize=9, color=SINK_COLOR, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=SINK_COLOR, linewidth=1.3),
    zorder=6,
)

ax.set_title(
    "Attention mask: frame-index columns, numbers = RoPE position\n"
    f"(sink f0 plotted at last_frame + {LOOKAHEAD_DISTANCE};  non-sink at actual frame idx)",
    fontsize=13, pad=40,
)

# Legend
ax.legend(
    handles=[
        mpatches.Patch(facecolor=NONSINK_COLOR, label="non-sink attends"),
        mpatches.Patch(facecolor=SINK_COLOR, label="sink attends (virtual future col)"),
        mpatches.Patch(facecolor=VACATED_COLOR, edgecolor="#aaa",
                       label="sink's vacated natural slot"),
        mpatches.Patch(facecolor=SKIPPED_BLOCK_BG, edgecolor="#aaa",
                       label="skipped block (gap in window)"),
        mpatches.Patch(facecolor=EMPTY_COLOR, edgecolor="#aaa", label="masked / not in window"),
    ],
    loc="lower center", bbox_to_anchor=(0.5, -0.20),
    ncol=3, frameon=False, fontsize=9.5,
)

fig.text(
    0.5, 0.005,
    "Columns = actual frame indices (sink plotted at last_frame + lookahead). "
    "Cell numbers = RoPE position applied to that key during that chunk's attention.\n"
    "Non-sink RoPE positions go 1..F_w-1 in window order (so f3 at col 3 gets pos 3 "
    "during C1 but pos 1 during C3 after f1/f2 eviction).  Sink gets F_w-1+"
    f"{LOOKAHEAD_DISTANCE}.",
    ha="center", va="bottom", fontsize=9.5,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8e1", edgecolor="#c9b458"),
)

plt.tight_layout(rect=(0, 0.04, 1, 1))
out2 = OUT_DIR / "attn_map_w9s1_rope.png"
plt.savefig(out2, dpi=160, bbox_inches="tight")
print(f"saved {out2}")
plt.close(fig)
