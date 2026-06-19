import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(np):
    np.arange(10)
    return


@app.cell
def _(np):
    np.linspace(0,1, 10)
    return


@app.cell
def _(np):
    np.zeros([3,3])
    return


@app.cell
def _(np):
    np.ones([3,3])
    return


@app.cell
def _(np):
    sf = 256
    t = np.arange(0,1,1/sf)
    print(len(t))
    print(t[:3], t[-3:])
    return sf, t


@app.cell
def _(np):
    a3 = np.arange(12)
    a4 = a3.reshape(3,4)
    a5 = a4.reshape(4,3)
    print(a3, a4,a5.T)
    return


@app.cell
def _(np):
    b1 = np.arange(20)
    print(b1[:5], b1[-5:])
    print(b1[5:-5])
    print(b1[::-1])
    return


@app.cell
def _(np):
    b2 = np.arange(16).reshape(4,4)
    print(b2[1,])
    print(b2[:,-1])
    print(b2[:2,:2])
    print(np.diag(b2))
    return


@app.cell
def _(np, sf):
    sig = np.arange(sf)
    sig[int(0.25*sf): int(0.75*sf)]
    return


@app.cell
def _(np):
    M = np.arange(12).reshape(3,4)
    return (M,)


@app.cell
def _(M):
    print(M-M.mean(axis=0))
    return


@app.cell
def _(np):
    C2 = np.arange(1,11).reshape(-1,1) * np.arange(1,11)
    return (C2,)


@app.cell
def _(C2):
    print(C2)
    return


@app.cell
def _(np):
    data = np.random.randn(4, 256)
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    normed = (data - mean) / std
    print(normed)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###### D1
    """)
    return


@app.cell
def _(np):
    sum_of_squares = (np.arange(1000)**2).sum()
    print(sum_of_squares)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ######D2
    """)
    return


@app.cell
def _(np):
    x = np.linspace(0,2*np.pi, 100)
    sin_x = np.sin(x)
    print(sin_x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###### D3
    """)
    return


@app.cell
def _(np):
    a = np.array([1, 2, 3])
    b = np.array([4, 6, 8])
    print(np.sqrt(np.sum((a-b)**2)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###### D4
    """)
    return


@app.cell
def _(np):
    s = np.random.randn(256)
    print(np.where(s <0, 0,s)) 
    print(s * (s > 0))
    return


@app.cell
def _(np, plt):
    #F1. Plot a 10 Hz sine wave over 1 second (sampled at 256 Hz). Add x-label "Time (s)", y-label "Amplitude", and a title.
    sf = 256
    t = np.arange(0,1,1/sf)


    plt.plot(t, np.sin(2*np.pi*10*t))
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("10 Hz sine")
    plt.show()
    return sf, t


@app.cell
def _(np, plt, t):
    #F2. Plot three sine waves — 5 Hz, 10 Hz, and 20 Hz — on the same axes, each with a label, and show a legend.
    for f in (5, 10, 20):
        plt.plot(t, np.sin(2*np.pi*f*t), label=f"{f} Hz")
    plt.legend()
    plt.show()
    return


@app.cell
def _(np, plt, t):
    #F3. Same three waves, but in three stacked subplots (one per wave), sharing the x-axis.
    fig, ax = plt.subplots(3, 1, sharex=True)
    for i, f3 in enumerate((5, 10, 20)):
        ax[i].plot(t, np.sin(2*np.pi*f3*t))
        ax[i].set_ylabel(f"{f3} Hz")
    plt.show()
    return


@app.cell
def _(np, plt, t):
    #F4. Make a clean 10 Hz sine, then a noisy version (+ 0.4 * np.random.randn(...)), and plot both overlaid so you can see the noise.

    clean = np.sin(2*np.pi*10*t)
    noisy = clean + 0.4 * np.random.randn(len(t))
    plt.plot(t, noisy, label="noisy")
    plt.plot(t, clean, label="clean")
    plt.legend()
    plt.show()
    return


@app.cell
def _(np, plt):
    #G1. The one-line signal. Build a signal that is the sum of sine waves at 5, 10, 20, and 40 Hz with amplitudes 1, 2, 0.5, 0.3 — using broadcasting, with no loop over frequencies. (Hint: make frequencies and amplitudes column vectors of shape (4,1), time a row vector of shape (1,N), and let broadcasting build a (4,N) array you then sum.)

    sf = 256
    t = np.arange(0, 2, 1/sf)

    freqs = np.array([5, 10, 20, 40]).reshape(-1, 1)
    amps = np.array([1, 2, 0.5, 0.3]).reshape(-1, 1)
    tt = t.reshape(1, -1)
    signal = (amps * np.sin(2*np.pi*freqs*tt)).sum(axis=0)
    print(signal)
    plt.plot(t,signal)
    plt.show()
    return sf, signal, t


@app.cell
def _(plt, signal, t):
    #G2. Downsample. Take a 256 Hz signal and downsample it to 128 Hz by keeping every other sample. Plot original vs downsampled on shared axes.

    down = signal[::2]
    plt.plot(t, signal, label="256 Hz")
    plt.plot(t[::2], down, label="128 Hz")
    plt.legend()
    plt.show()
    return


@app.cell
def _(np, plt, signal, t):
    #G3. Smooth it. Apply a moving-average smoother of window 5 to a noisy signal — no explicit loop. (Hint: np.convolve.)
    smooth = np.convolve(signal, np.ones(5)/5, mode='same')
    plt.plot(t, smooth)
    plt.show()
    return


@app.cell
def _(np):
    #G4. Find the loudest frequency. Given amps = np.array([0.2, 5.0, 1.1, 0.3]) for frequencies [5,10,20,40], return the frequency with the largest amplitude — no loop.

    freqs_g4 = np.array([5, 10, 20, 40])
    amps_g4 = np.array([0.2, 5.0, 1.1, 0.3])
    print(freqs_g4[np.argmax(amps_g4)])
    return


@app.cell
def _(np, plt):
    #Write a fully vectorized function (no loops) that you'll genuinely reuse in later weeks:
    """
    def make_eeg(freqs, amps, sf=256, duration=2, noise=0.3):
        '''
        Build a synthetic multi-frequency EEG-like signal.
        freqs, amps : lists/arrays of equal length
        returns: (t, signal)
        '''
        # your code here — use broadcasting, no for loops

    Then call it with freqs=[10, 20], amps=[2, 0.5] and plot the result with proper labels. If your alpha (10 Hz) component visibly dominates, you've nailed Week 1.
    """
    def make_eeg(freqs, amps, sf=256, duration=2, noise=0.3):
        '''
        Build a synthetic multi-frequency EEG-like signal.
        freqs, amps : lists/arrays of equal length
        returns: (t, signal)
        '''
        t = np.arange(0, duration, 1/sf)
        freqs = np.array(freqs).reshape(-1, 1)
        amps = np.array(amps).reshape(-1, 1)
        components = amps * np.sin(2 * np.pi * freqs * t.reshape(1, -1))
        signal = components.sum(axis=0) + noise * np.random.randn(len(t))
        return t, signal

    t_cap, sig_cap = make_eeg([10, 20], [2, 0.5])
    plt.plot(t_cap, sig_cap)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Synthetic EEG (alpha-dominant)")
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
