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
    mo.md("""
    ### Set D
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set F
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(np, plt):
    # F1
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
    # F2
    for f in (5, 10, 20):
        plt.plot(t, np.sin(2*np.pi*f*t), label=f"{f} Hz")
    plt.legend()
    plt.show()
    return


@app.cell
def _(np, plt, t):
    # F3
    fig, ax = plt.subplots(3, 1, sharex=True)
    for i, f3 in enumerate((5, 10, 20)):
        ax[i].plot(t, np.sin(2*np.pi*f3*t))
        ax[i].set_ylabel(f"{f3} Hz")
    plt.show()
    return


@app.cell
def _(np, plt, t):
    #F4

    clean = np.sin(2*np.pi*10*t)
    noisy = clean + 0.4 * np.random.randn(len(t))
    plt.plot(t, noisy, label="noisy")
    plt.plot(t, clean, label="clean")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
