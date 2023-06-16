import sys

sys.path.append("/Users/yafo/Library/Mobile Documents/com~apple~CloudDocs/IDC MLDS MSc 2021/1_Python/0_MLDS.Fall_TASHPAG/Numerical_Optimization_3327/HW_prog/")

import unittest
import numpy as np
from src.utils import contour, plot
from src.unconstrained_min import LineSearchSolver
from tests.examples import (
    Q1_circle,
    Q2_ellipse,
    Q3_rot_ellipse,
    Rosenbrock,
    linear,
    exp_tri,
)

class TestLineSearchMinMethods(unittest.TestCase):
    x0 = np.array([1, 1])
    x0_rosen = np.array([-1, 2], dtype=np.float64)
    gd_minimizer = LineSearchSolver("gradient descent")
    newton_minimizer = LineSearchSolver("Newton")
    bfgs_minimizer = LineSearchSolver("BFGS")
    sr1_minimizer = LineSearchSolver("sr1")
    
    def test1(self):
        f, ot, pt, iters = Q1_circle, 10e-8, 10e-12, 100

        self.gd_minimizer.minimize(f, self.x0, ot, pt, iters)
        self.newton_minimizer.minimize_newton(f, self.x0, 0.001, 0.001, iters)
        self.bfgs_minimizer.minimize(f, self.x0, ot, pt, iters)
        self.sr1_minimizer.minimize(f, self.x0, ot, pt, iters)
        
        contour(f, self.gd_minimizer.x_hist, self.newton_minimizer.x_hist, self.bfgs_minimizer.x_hist, self.sr1_minimizer.x_hist, [-2.5, 2.5], [-2.5, 2.5], "Example d.i. Contour $x_{1}^{2} + x_{2}^{2}$")
        plot(self.gd_minimizer.f_x_hist, self.newton_minimizer.f_x_hist, self.bfgs_minimizer.f_x_hist, self.sr1_minimizer.f_x_hist, "Example d.i. Plot $x_{1}^{2} + x_{2}^{2}$")

    def test2(self):
        f, ot, pt, iters = Q2_ellipse, 10e-8, 10e-12, 100
        
        self.gd_minimizer.minimize(f, self.x0, ot, pt, iters)
        self.newton_minimizer.minimize_newton(f, self.x0, 0.001, 0.001, iters)
        self.bfgs_minimizer.minimize(f, self.x0, ot, pt, iters)
        self.sr1_minimizer.minimize(f, self.x0, ot, pt, iters)
        
        contour(f, self.gd_minimizer.x_hist, self.newton_minimizer.x_hist, self.bfgs_minimizer.x_hist, self.sr1_minimizer.x_hist, [-2.5, 2.5], [-2.5, 2.5], "Example d.ii. Contour $x_{1}^{2} + 100x_{2}^{2}$")
        plot(self.gd_minimizer.f_x_hist, self.newton_minimizer.f_x_hist, self.bfgs_minimizer.f_x_hist, self.sr1_minimizer.f_x_hist, "Example d.ii. Plot $x_{1}^{2} + 100x_{2}^{2}$")

    def test3(self):
        f, ot, pt, iters = Q3_rot_ellipse, 10e-8, 10e-12, 100
        
        self.gd_minimizer.minimize(f, self.x0, ot, pt, iters)
        self.newton_minimizer.minimize_newton(f, self.x0, 0.001, 0.001, iters)
        self.bfgs_minimizer.minimize(f, self.x0, ot, pt, iters)
        self.sr1_minimizer.minimize(f, self.x0, ot, pt, iters)
        
        contour(f, self.gd_minimizer.x_hist, self.newton_minimizer.x_hist, self.bfgs_minimizer.x_hist, self.sr1_minimizer.x_hist, [-2.5, 2.5], [-2.5, 2.5], "Example d.iii. Contour $100x_{1}^{2} + x_{2}^{2}$, rotated 30 degrees")
        plot(self.gd_minimizer.f_x_hist, self.newton_minimizer.f_x_hist, self.bfgs_minimizer.f_x_hist, self.sr1_minimizer.f_x_hist, "Example d.iii. Plot $100x_{1}^{2} + x_{2}^{2}$, rotated 30 degrees")


    def test4(self):
        f, ot, pt, iters = Rosenbrock, 10e-8, 10e-12, 10000
        
        self.gd_minimizer.minimize(f, self.x0_rosen, ot, pt, iters)
        self.newton_minimizer.minimize_newton(f, self.x0_rosen, 0.001, 0.001, iters)
        self.bfgs_minimizer.minimize(f, self.x0_rosen, ot, pt, iters)
        self.sr1_minimizer.minimize(f, self.x0_rosen, ot, pt, iters)
        
        contour(f, self.gd_minimizer.x_hist, self.newton_minimizer.x_hist, self.bfgs_minimizer.x_hist, self.sr1_minimizer.x_hist, [-2.5, 2.5], [-2.5, 2.5], "Example e. Contour Rosenbrock 2D")
        plot(self.gd_minimizer.f_x_hist, self.newton_minimizer.f_x_hist, self.bfgs_minimizer.f_x_hist, self.sr1_minimizer.f_x_hist, "Example e. Plot Rosenbrock 2D")


    def test5(self):
        f, ot, pt, iters = linear, 10e-8, 10e-12, 100
        
        self.gd_minimizer.minimize(f, self.x0, ot, pt, iters)
        
        contour(f, self.gd_minimizer.x_hist, [], [], [], [-200, 2], [-200, 2], "Example f. Contour $f(x)=a^{T}x$")
        plot(self.gd_minimizer.f_x_hist, [], [], [], "Example f. Plot $f(x)=a^{T}x$")
        

    def test6(self):
        f, ot, pt, iters = exp_tri, 10e-8, 10e-12, 100
        
        self.gd_minimizer.minimize(f, self.x0_rosen, ot, pt, iters)
        self.newton_minimizer.minimize_newton(f, self.x0_rosen, 0.001, 0.001, iters)
        self.bfgs_minimizer.minimize(f, self.x0_rosen, ot, pt, iters)
        self.sr1_minimizer.minimize(f, self.x0_rosen, ot, pt, iters)
        
        contour(f, self.gd_minimizer.x_hist, self.newton_minimizer.x_hist, self.bfgs_minimizer.x_hist, self.sr1_minimizer.x_hist, [-1.3, 2.5], [-3, 3], "Example g. Contour $f(x_{1},x_{2}) = e^{x_{1} + 3x_{2} - 0.1} + e^{x_{1} - 3x_{2} - 0.1} + e^{-x_{1} - 0.1}$")
        plot(self.gd_minimizer.f_x_hist, self.newton_minimizer.f_x_hist, self.bfgs_minimizer.f_x_hist, self.sr1_minimizer.f_x_hist, "Example g. Plot $f(x_{1},x_{2}) = e^{x_{1} + 3x_{2} - 0.1} + e^{x_{1} - 3x_{2} - 0.1} + e^{-x_{1} - 0.1}$")


