import sympy as sym
from sympy import *
from sympy.plotting import plot
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np

init_printing()

'''
Define Variable Equations to Solve'''
equation_a = "(x**2+y)*(y**2+x)"
equation_b = "3*x**2 + y**2"
equation_c = "sin(theta) + 2*cos(phi)"
equations = [equation_a, equation_b, equation_c]


'''
Equations Class to effect Variable Equations
'''


class Equation:
    def __init__(self, form):
        super().__init__()
        self.form = sympify(form)
        self.expanded = sym.expand(sympify(form))
        self.symbols = list(self.form.free_symbols)

    def expand(self):
        print("Expanded: \n", self.expanded)
        return self.expanded

    def latex(self):
        print("Latex Form: \n", latex(self.form))
        return latex(self.form)

    def numpy(self):
        numpy = lambdify(self.symbols, self.form, 'numpy')
        print("Numpy Form Saved Using Variables: \n", self.symbols)
        return numpy, self.symbols

    def plot(self):
        plot_a = plot(self.form, show=True)
        return plot_a


equation_objs = []
for equation in equations:
    equation_objs.append(Equation(equation))


'''
Functions on Variable Equations
'''
np_objs = {}
equation_variables = {}
for equation_obj in equation_objs:
    print("For Equation: ", equations[equation_objs.index(equation_obj)])
    equation_expanded = equation_obj.expand()
    equation_latex = equation_obj.latex()
    equation_numpy, equation_vars = equation_obj.numpy()
    np_objs[equations[equation_objs.index(equation_obj)]] = equation_numpy
    equation_variables[equations[equation_objs.index(equation_obj)]] = equation_vars
    print("\n")

