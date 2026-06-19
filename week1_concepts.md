# Week 1 — NumPy & Matplotlib Concepts (Learning Guide)

A companion to [practice_w1.py](practice_w1.py) and the drill pack in [week1.md](week1.md).
This explains *what* each idea is, *why* it matters for signal/EEG work, and *how* to
memorize it and build intuition.

The whole week builds toward one thing: **make a signal, look at it, and manipulate it without writing loops.** Everything below is a stepping stone to that.

---

## 0. The one mental model that ties it all together

A signal is just **an array of numbers sampled over time**.

- `sf` (sampling frequency) = how many samples per second (256 Hz = 256 numbers per second).
- A time vector `t` is the x-axis (seconds). A signal is the y-axis (amplitude).
- "Doing signal processing" = doing array math on `t` and the signal.

If you remember only one thing: **NumPy lets you apply math to a whole array at once, no loop needed.** `np.sin(t)` takes the sine of *every* element. That's the superpower the whole week trains.

---

## 1. Making arrays (the raw material)

| Code | What it gives you | When you use it |
|------|------------------|-----------------|
| `np.arange(10)` | `[0 1 2 ... 9]` — integers, like Python `range` | counting, indices |
| `np.linspace(0, 1, 10)` | 10 evenly-spaced numbers *including both ends* | x-axes, time when you want an exact count |
| `np.zeros([3,3])` | 3×3 of zeros | empty buffer to fill in |
| `np.ones([3,3])` | 3×3 of ones | smoothing kernels, masks |

**arange vs linspace — the #1 beginner confusion:**
- `arange(start, stop, step)` → you control the **step size**, stop is *excluded*.
- `linspace(start, stop, num)` → you control the **count**, stop is *included*.

> 🧠 **Trick to remember:** ar**a**nge = "**a**dd this much each time" (step). lin**space** = "give me this many e**venly s**paced points" (count).

**The time-vector pattern** (you'll write this 100 times):
```python
sf = 256
t = np.arange(0, 1, 1/sf)   # 1 second, 256 samples, step = 1/256
```
`len(t)` is 256, not 257, because the stop value (1.0) is excluded. That "off-by-one" is normal and correct.

---

## 2. Shape: reshape & transpose

An array has a **shape** — `(3, 4)` means 3 rows, 4 columns.

- `a.reshape(3, 4)` rearranges 12 numbers into 3×4 (must multiply to the same total).
- `.reshape(-1, 1)` means "1 column, you figure out the rows" → a **column vector**. The `-1` = "infer this dimension." This is the single most useful trick of the week.
- `.T` transposes (flips rows ↔ columns).

> 🧠 **Intuition:** reshape never changes the *numbers* or their order, only how they're *boxed*. Think of pouring the same beads into a differently-shaped tray.

---

## 3. Slicing (grabbing pieces) — the real day-to-day skill

Slicing is `array[start:stop:step]`, and `stop` is always **excluded**.

**1D:**
```python
b[:5]      # first 5
b[-5:]     # last 5
b[5:-5]    # middle (drop 5 from each end)
b[::-1]    # reversed (step of -1)
b[::2]     # every other element  → this is downsampling!
```

**2D** is the same idea with a comma: `array[rows, cols]`
```python
b2[1, ]      # row 1 (whole row)
b2[:, -1]    # last column (: = all rows)
b2[:2, :2]   # top-left 2×2 block
np.diag(b2)  # the diagonal
```

> 🧠 **Trick:** read the colon as "**through**" and the comma as "**then**." `b2[:2, :2]` = "rows up-to-2, then cols up-to-2."

**Why neuro cares:** grabbing a time window is just slicing.
```python
sig[int(0.25*sf) : int(0.75*sf)]   # the chunk from 0.25 s to 0.75 s
```

---

## 4. Vectorized math & broadcasting (the heart of the week)

**Vectorized** = apply an operation to the whole array at once:
```python
(np.arange(1000)**2).sum()    # sum of squares, no loop
np.sin(x)                     # sine of every point
np.sqrt(np.sum((a-b)**2))     # Euclidean distance in one line
```

**Broadcasting** = NumPy automatically stretches a smaller array to match a bigger one so they can combine. The rule of thumb: a dimension of size 1 gets *stretched* to fit.

The killer example from G1 — build 4 sine waves with **no loop over frequencies**:
```python
freqs = np.array([5, 10, 20, 40]).reshape(-1, 1)  # shape (4, 1)  column
amps  = np.array([1, 2, 0.5, 0.3]).reshape(-1, 1) # shape (4, 1)  column
tt    = t.reshape(1, -1)                            # shape (1, N)  row

signal = (amps * np.sin(2*np.pi*freqs*tt)).sum(axis=0)
```
Here `(4,1)` times `(1,N)` broadcasts into a `(4,N)` grid — one row per frequency — and `.sum(axis=0)` collapses the 4 rows into one combined signal.

> 🧠 **Intuition:** column (vertical) × row (horizontal) → a full grid (table). Think of a multiplication table: `np.arange(1,11).reshape(-1,1) * np.arange(1,11)` builds exactly that. If you can picture the times-table, you understand broadcasting.

**`axis` — the other thing beginners trip on:**
- `axis=0` = collapse **down the rows** (result has one value per column).
- `axis=1` = collapse **across the columns** (result has one value per row).
- `keepdims=True` keeps the result 2D so it broadcasts cleanly back:
```python
mean = data.mean(axis=1, keepdims=True)   # one mean per row, kept as a column
normed = (data - mean) / std              # z-score each row
```
> 🧠 **Trick:** `axis` is the dimension that *disappears*. `data` is `(4, 256)`; `mean(axis=1)` removes the 256 → you get 4 means.

---

## 5. Masks & conditions (selecting by value)

```python
np.where(s < 0, 0, s)   # "where s<0 put 0, else keep s"  (this is ReLU / clipping)
s * (s > 0)             # same result: True=1, False=0, so negatives become 0
np.argmax(amps)         # index of the largest value
freqs[np.argmax(amps)]  # the frequency with the biggest amplitude
```

> 🧠 **Intuition:** a comparison like `s > 0` returns an array of `True/False`. You can use that as a filter (a "mask") or multiply by it (True=1, False=0). `argmax` answers "**where** is the max?" not "**what** is the max?"

---

## 6. Matplotlib — looking at your signal

The basic recipe never changes:
```python
plt.plot(t, np.sin(2*np.pi*10*t))   # x, then y
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("10 Hz sine")
plt.show()
```
> 🧠 A 10 Hz sine is `sin(2 * π * f * t)` with `f=10`. **2π is what turns "cycles per second" into the radians `sin` wants.** Memorize `2*np.pi*f*t` as one chunk — it's the formula for every wave you'll make.

**Multiple lines + legend:**
```python
for f in (5, 10, 20):
    plt.plot(t, np.sin(2*np.pi*f*t), label=f"{f} Hz")
plt.legend()        # legend only shows up if you set label= on each plot
```
> ⚠️ A loop over a *handful of plot calls* is fine — the "no loops" rule is about looping over the **data array**, not over 3 lines you want to draw.

**Subplots (stacked panels):**
```python
fig, ax = plt.subplots(3, 1, sharex=True)  # 3 rows, 1 col, shared x-axis
ax[0].plot(...); ax[0].set_ylabel("5 Hz")
```
> 🧠 **One plot vs subplots:** use `plt.plot` to overlay lines on *one* axes; use `plt.subplots` when you want *separate* panels. With subplots you call methods on `ax` (`ax.plot`, `ax.set_ylabel`) instead of `plt.`.

**Seeing noise** (a recurring neuro task):
```python
clean = np.sin(2*np.pi*10*t)
noisy = clean + 0.4 * np.random.randn(len(t))   # add random noise
plt.plot(t, noisy, label="noisy")
plt.plot(t, clean, label="clean")
```

---

## 7. Three signal-processing moves you now own

1. **Downsample** — keep every Nth sample: `signal[::2]` (256 Hz → 128 Hz). Remember to also downsample the time axis: `t[::2]`.
2. **Smooth** — moving average via convolution, no loop:
   ```python
   smooth = np.convolve(signal, np.ones(5)/5, mode='same')
   ```
   `np.ones(5)/5` is a window of five 0.2's = "average of 5 neighbors." `mode='same'` keeps the output the same length as the input.
3. **Find the dominant frequency** — `freqs[np.argmax(amps)]`.

---

## 8. The capstone — proof you "get" Week 1

```python
def make_eeg(freqs, amps, sf=256, duration=2, noise=0.3):
    t = np.arange(0, duration, 1/sf)
    freqs = np.array(freqs).reshape(-1, 1)      # column
    amps  = np.array(amps).reshape(-1, 1)       # column
    components = amps * np.sin(2*np.pi*freqs * t.reshape(1, -1))  # broadcast → (F, N)
    signal = components.sum(axis=0) + noise * np.random.randn(len(t))
    return t, signal
```
This single function uses **every** Week 1 skill: time vector, reshape to columns, broadcasting, `sum(axis=0)`, noise, and it returns data you then plot. If the 10 Hz (alpha) component visibly dominates when amps favor it, you've nailed it.

---

## How to actually memorize this & build intuition

**Don't memorize syntax — memorize a handful of patterns.** There are really only ~6:

1. Time vector: `t = np.arange(0, dur, 1/sf)`
2. A wave: `np.sin(2*np.pi*f*t)`
3. Column vector: `.reshape(-1, 1)`
4. Collapse a dimension: `.sum(axis=0)` / `.mean(axis=1)`
5. Slice a window / downsample: `x[a:b]`, `x[::2]`
6. Plot recipe: `plot → xlabel → ylabel → title → show`

**Tips that make it stick:**

- **Type it, don't read it.** Re-typing the 6 patterns from a blank file beats re-reading them 10×. Aim to write `make_eeg` from memory by end of week.
- **Always `print(arr.shape)`** when confused. 90% of NumPy bugs are shape bugs; seeing `(4, 256)` instantly tells you what `axis` to use.
- **Predict, then run.** Before running a cell, say out loud what shape/value you expect. Being wrong is how the intuition forms (marimo re-runs instantly, so abuse that feedback loop).
- **Reach for "no loop."** Whenever you start typing `for`, pause and ask "can broadcasting do this?" That instinct *is* the skill these exercises build.
- **Build the smallest version first.** Make a `(2,3)` array and slice it by hand before trusting a `(4,256)` one. Small arrays you can verify with your eyes.
- **Keep a personal cheat-sheet** of the 6 patterns above. Glance, don't google.

**Spaced repetition:** there's a `week1_anki.txt` in this repo — use it. Reviewing 5 minutes a day for a week beats one long cram session. Re-do the D and G sets from [week1.md](week1.md) from a blank cell on day 3 and day 7; if you can write `make_eeg` cold, Week 1 is done.

---

### Quick glossary
- **sf / sampling frequency** — samples per second (Hz).
- **vectorized** — operating on a whole array at once, no Python loop.
- **broadcasting** — auto-stretching a size-1 dimension so arrays of different shapes combine.
- **axis** — the dimension an operation collapses (`0` = down rows, `1` = across cols).
- **mask** — a boolean array used to select or zero-out elements.
- **convolution** — sliding a small window across a signal (used here to smooth).
