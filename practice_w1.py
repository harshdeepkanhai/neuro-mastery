import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    return


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
    return (sf,)


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
