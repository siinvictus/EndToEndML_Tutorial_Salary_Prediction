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


@app.function
def measure_error(pred, groundtruth):
    error_list = []
    for i in range(len(pred)):
        error = pred[i] - groundtruth[i]
        error_list.append(error)
    return error_list


@app.function
def relu_derivative(input):
    if input>0:
        return 1
    else:
        return 0


@app.cell
def _(a2, act_output1, output1, output2, x):
    def backward_pass(errors):
        # backward pass
        delta2 = 2 * errors * relu_derivative(output2)
        delta1 = delta2 * a2 * relu_derivative(output1)
    
        grad_a2 = delta2 * act_output1
        grad_b2 = delta2
    
        grad_a1 = delta1 * x
        grad_b1 = delta1
    
        return grad_a1, grad_b1, grad_a2, grad_b2
    

    return


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


@app.cell
def _():
    my_l = { 'age': [20,23,21,30,40,33,18,47,52],
             'years_edu': [2,5,3,6,6,5,1,6,6]
           }
    return (my_l,)


@app.cell
def _(my_l):
    predictions_forward = network_just_forward(1,2,3,4, my_l['age'])
    predictions_forward
    return (predictions_forward,)


@app.cell
def _(my_l, predictions_forward):
    measure_error(predictions_forward, my_l['years_edu'])
    return


if __name__ == "__main__":
    app.run()
