import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # <center> 1. Neural Network Manual
    """)
    return


@app.function
def make_neuron(a,b,input):
    output = a*input + b
    return output


@app.function
def make_activation(input):
    if input<0:
        output = 0
    else:
        output = input
    return output


@app.cell
def _():
    output1= make_neuron(1,2,10)
    return (output1,)


@app.cell
def _(output1):
    act1_output = make_activation(output1)
    act1_output
    return (act1_output,)


@app.cell
def _(act1_output):
    output2 = make_neuron(1,2,act1_output)

    return (output2,)


@app.cell
def _(output2):
    act2_output = make_activation(output2)
    act2_output
    return


@app.function
def network_just_forward(a1,b1,a2,b2, x):
    predictions = []
    for el in x:
        output1 = make_neuron(a1,b1,el)
        act_output1 = make_activation(output1)
        output2 = make_neuron(a2,b2,act_output1)
        act_output2 = make_activation(output2)
        predictions.append(act_output2)
    return predictions


@app.cell
def _():
    my_l = [20,23,21,30,40,33,18,47,52]
    sorted_l = sorted(my_l)
    print(sorted_l)
    return (sorted_l,)


@app.cell
def _(sorted_l):
    network_just_forward(1,2,3,4,sorted_l)
    return


if __name__ == "__main__":
    app.run()
