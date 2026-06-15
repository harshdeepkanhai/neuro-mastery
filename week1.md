# Week 1 Mastery Pack — NumPy + Matplotlib Drills & Puzzles

Everything here uses only Week 1 skills: arrays, indexing, slicing, broadcasting, vectorization (no `for` loops over data), boolean masking, aggregations, and Matplotlib. Every exercise is EEG-flavored so the skills carry straight into later weeks.

**How to use this:**

1. Open a fresh Jupyter notebook. Start with `import numpy as np` and `import matplotlib.pyplot as plt`.
2. Try each one yourself first. Struggle a little — that's the learning.
3. Only then check the **Solutions** section at the bottom.
4. ⭐ = drill (build the muscle) · 🧩 = puzzle (think harder)
5. **Hard rule for the whole pack: no `for` loops over your data. If you reach for a loop, find the vectorized way.**

---

## Set A — Array Basics ⭐

**A1.** Create each of these in one line:

- the integers 0 through 9
- 10 evenly spaced numbers from 0 to 1 (inclusive)
- a 3×3 array of all zeros, and one of all ones
- a 3×3 identity matrix

**A2.** Build a **time vector** for 1 second of EEG sampled at 256 Hz. How many samples does it contain? Print the length, and the first and last 3 values.

**A3.** Make `np.arange(12)`, reshape it to 3×4, then to 4×3, then transpose the 4×3 back. What shape do you end with?

---

## Set B — Indexing & Slicing ⭐

**B1.** From `np.arange(20)`, extract: the first 5 elements, the last 5, every other element, and the whole array reversed.

**B2.** Make a 4×4 array with `np.arange(16).reshape(4,4)`. Extract:

- the second row
- the last column
- the top-left 2×2 block
- the main diagonal

**B3.** A signal `sig = np.arange(256)` represents 1 second at 256 Hz. Slice out just the **middle half-second** (samples from 0.25 s to 0.75 s).

---

## Set C — Broadcasting 🧩

This is the concept people find hardest. Master it now and the rest of EEG gets easy.

**C1.** Given `M = np.arange(12).reshape(3,4)`, subtract the **mean of each column** from that column — so each column ends up with mean 0. (One line. No loop.)

**C2.** 🧩 Build a 10×10 multiplication table (entry `[i,j] = (i+1)*(j+1)`) in a single line using broadcasting. No loops.

**C3.** 🧩 You have a fake EEG array `data` of shape (4 channels, 256 samples): `data = np.random.randn(4, 256)`. Normalize **each channel** to zero mean and unit standard deviation — vectorized, no loop. (Hint: `keepdims=True` is your friend.)

---

## Set D — Vectorization (kill the loop) ⭐🧩

**D1.** Rewrite this with no loop:

```python
total = 0
for x in range(1000):
    total += x**2
```

**D2.** For `x = np.linspace(0, 2*np.pi, 100)`, compute `sin(x)` for every element at once.

**D3.** 🧩 Given `a = np.array([1,2,3])` and `b = np.array([4,6,8])`, compute the Euclidean distance between them with no loop.

**D4.** Given `s = np.random.randn(256)`, apply a "ReLU" — replace every negative value with 0 — without a loop.

---

## Set E — Boolean Masking & Aggregation ⭐🧩

**E1.** For `s = np.random.randn(1000)`: count how many values are greater than 1, and pull out all values whose absolute value exceeds 2 (the "outliers").

**E2.** For `data = np.random.randn(4, 256)` (4 channels): compute the mean and standard deviation of **each channel**, then find the **index of the channel with the highest variance**.

**E3.** 🧩 For a signal `s`, find all the indices where it **crosses zero** (changes sign). This is a real trick used to estimate frequency. (Hint: `np.sign` and `np.diff`.)

---

## Set F — Matplotlib ⭐

**F1.** Plot a 10 Hz sine wave over 1 second (sampled at 256 Hz). Add x-label "Time (s)", y-label "Amplitude", and a title.

**F2.** Plot three sine waves — 5 Hz, 10 Hz, and 20 Hz — on the **same axes**, each with a label, and show a legend.

**F3.** Same three waves, but in **three stacked subplots** (one per wave), sharing the x-axis.

**F4.** Make a clean 10 Hz sine, then a noisy version (`+ 0.4 * np.random.randn(...)`), and plot both overlaid so you can see the noise.

---

## Set G — Integrative Puzzles 🧩

**G1. The one-line signal.** Build a signal that is the sum of sine waves at **5, 10, 20, and 40 Hz** with amplitudes **1, 2, 0.5, 0.3** — using broadcasting, with **no loop over frequencies**. (Hint: make frequencies and amplitudes column vectors of shape (4,1), time a row vector of shape (1,N), and let broadcasting build a (4,N) array you then sum.)

**G2. Downsample.** Take a 256 Hz signal and downsample it to 128 Hz by keeping every other sample. Plot original vs downsampled on shared axes.

**G3. Smooth it.** Apply a moving-average smoother of window 5 to a noisy signal — no explicit loop. (Hint: `np.convolve`.)

**G4. Find the loudest frequency.** Given `amps = np.array([0.2, 5.0, 1.1, 0.3])` for frequencies `[5,10,20,40]`, return the frequency with the largest amplitude — no loop.

---

## Capstone Challenge 🏆

Write a fully vectorized function (no loops) that you'll genuinely reuse in later weeks:

```python
def make_eeg(freqs, amps, sf=256, duration=2, noise=0.3):
    """
    Build a synthetic multi-frequency EEG-like signal.
    freqs, amps : lists/arrays of equal length
    returns: (t, signal)
    """
    # your code here — use broadcasting, no for loops
```

Then call it with `freqs=[10, 20], amps=[2, 0.5]` and plot the result with proper labels. If your alpha (10 Hz) component visibly dominates, you've nailed Week 1.

---

---

# SOLUTIONS

_(Try everything above before reading. Seriously.)_

### Set A

```python
# A1
np.arange(10)
np.linspace(0, 1, 10)
np.zeros((3,3)); np.ones((3,3)); np.eye(3)

# A2
sf = 256
t = np.arange(0, 1, 1/sf)
print(len(t))          # 256
print(t[:3], t[-3:])

# A3
a = np.arange(12).reshape(3,4)
b = a.reshape(4,3)
c = b.T                # shape (3,4)
```

### Set B

```python
x = np.arange(20)
x[:5]; x[-5:]; x[::2]; x[::-1]          # B1

m = np.arange(16).reshape(4,4)          # B2
m[1]            # second row
m[:, -1]        # last column
m[:2, :2]       # top-left 2x2
np.diag(m)      # diagonal

sig = np.arange(256)                    # B3
sig[64:192]     # 0.25s -> 0.75s  (0.25*256=64, 0.75*256=192)
```

### Set C

```python
M = np.arange(12).reshape(3,4)                 # C1
M - M.mean(axis=0)                             # column means broadcast across rows

np.arange(1,11).reshape(-1,1) * np.arange(1,11)  # C2  -> 10x10 table

data = np.random.randn(4, 256)                 # C3
mean = data.mean(axis=1, keepdims=True)
std  = data.std(axis=1, keepdims=True)
normed = (data - mean) / std
```

### Set D

```python
np.arange(1000)**2 .sum()  if written as:  (np.arange(1000)**2).sum()   # D1

x = np.linspace(0, 2*np.pi, 100); np.sin(x)                              # D2

a = np.array([1,2,3]); b = np.array([4,6,8])                            # D3
np.sqrt(np.sum((a-b)**2))     # = sqrt(9+16+25) = sqrt(50)

s = np.random.randn(256)                                                # D4
np.where(s < 0, 0, s)         # or: s * (s > 0)
```

### Set E

```python
s = np.random.randn(1000)                       # E1
(s > 1).sum()
s[np.abs(s) > 2]

data = np.random.randn(4, 256)                  # E2
data.mean(axis=1)
data.std(axis=1)
np.argmax(data.var(axis=1))                     # noisiest channel index

sign = np.sign(s)                               # E3
crossings = np.where(np.diff(sign) != 0)[0]
```

### Set F

```python
sf = 256; t = np.arange(0, 1, 1/sf)

# F1
plt.plot(t, np.sin(2*np.pi*10*t))
plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.title("10 Hz sine"); plt.show()

# F2
for f in (5,10,20):                 # (a loop over PLOT CALLS is fine — not over data)
    plt.plot(t, np.sin(2*np.pi*f*t), label=f"{f} Hz")
plt.legend(); plt.show()

# F3
fig, ax = plt.subplots(3, 1, sharex=True)
for i, f in enumerate((5,10,20)):
    ax[i].plot(t, np.sin(2*np.pi*f*t)); ax[i].set_ylabel(f"{f} Hz")
plt.show()

# F4
clean = np.sin(2*np.pi*10*t)
noisy = clean + 0.4*np.random.randn(len(t))
plt.plot(t, noisy, label="noisy"); plt.plot(t, clean, label="clean")
plt.legend(); plt.show()
```

_(Note: looping over a handful of plot calls is fine — the rule is no loops over the data array itself.)_

### Set G

```python
sf = 256; t = np.arange(0, 2, 1/sf)

# G1 — the one-line signal
freqs = np.array([5,10,20,40]).reshape(-1,1)     # (4,1)
amps  = np.array([1,2,0.5,0.3]).reshape(-1,1)    # (4,1)
tt    = t.reshape(1,-1)                           # (1,N)
signal = (amps * np.sin(2*np.pi*freqs*tt)).sum(axis=0)   # (N,)

# G2 — downsample
down = signal[::2]
plt.plot(t, signal, label="256 Hz")
plt.plot(t[::2], down, label="128 Hz"); plt.legend(); plt.show()

# G3 — smooth
smooth = np.convolve(signal, np.ones(5)/5, mode='same')

# G4 — loudest frequency
freqs = np.array([5,10,20,40]); amps = np.array([0.2,5.0,1.1,0.3])
freqs[np.argmax(amps)]      # -> 10
```

### Capstone

```python
def make_eeg(freqs, amps, sf=256, duration=2, noise=0.3):
    t = np.arange(0, duration, 1/sf)
    freqs = np.array(freqs).reshape(-1, 1)
    amps  = np.array(amps).reshape(-1, 1)
    components = amps * np.sin(2*np.pi*freqs*t.reshape(1, -1))
    signal = components.sum(axis=0) + noise*np.random.randn(len(t))
    return t, signal

t, sig = make_eeg([10, 20], [2, 0.5])
plt.plot(t, sig); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.title("Synthetic EEG (alpha-dominant)"); plt.show()
```

---

## Self-grading

You've mastered Week 1 if you can do, from a blank notebook without looking anything up:

- [ ] Build a time vector and a sine wave at any frequency
- [ ] Slice and index 1-D and 2-D arrays confidently
- [ ] Normalize per-channel using broadcasting + `keepdims`
- [ ] Replace any data loop with a vectorized expression
- [ ] Use boolean masks to filter and count
- [ ] Make labelled plots, multi-line plots, and subplots
- [ ] Write `make_eeg` from memory

When all boxes are checked, move to Week 2 (signal processing — the big one).
