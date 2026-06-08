import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## This is a test
    Ketu do shohim gjerat
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
