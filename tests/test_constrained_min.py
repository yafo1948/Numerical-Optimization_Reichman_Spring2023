import sys

sys.path.append("/Users/yafo/Library/Mobile Documents/com~apple~CloudDocs/IDC MLDS MSc 2021/1_Python/0_MLDS.Fall_TASHPAG/Numerical_Optimization_3327/HW_prog/")

import unittest
import numpy as np
from src.constrained_min import InteriorPointMinimizer
from src.utils import plot_iterations, plot_feasible_regions_3d, plot_feasible_set_2d
from tests.examples import (
    qp,
    ineq_constraint_1_1,
    ineq_constraint_1_2,
    ineq_constraint_1_3,
    lp,
    ineq_constraint_2_1,
    ineq_constraint_2_2,
    ineq_constraint_2_3,
    ineq_constraint_2_4,
)

class TestInteriorPointMethod(unittest.TestCase):
    starting_point_qp = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    starting_point_lp = np.array([0.5, 0.75], dtype=np.float64)
    minimizer = InteriorPointMinimizer()

    def test_qp(self):
        eq_constraint_mat = np.array([[1, 1, 1]]).reshape(1, -1)
        x_s, obj_values, outer_x_s, outer_obj_values = self.minimizer.minimize(
            qp,
            self.starting_point_qp,
            [ineq_constraint_1_1, ineq_constraint_1_2, ineq_constraint_1_3],
            eq_constraint_mat,
            np.array([1]),
            "wolfe",
            10e-12,
            10e-8,
            100,
            20,
            10e-10,
        )

        print(f"Point of convergence: {x_s[-1]}")
        print(f"Objective value at point of convergence: {qp(x_s[-1], False)[0]:.4f}")
        print(
            f"-x value at point of convergence: {ineq_constraint_1_1(x_s[-1], False)[0]:.4f}"
        )
        print(
            f"-y value at point of convergence: {ineq_constraint_1_2(x_s[-1], False)[0]:.4f}"
        )
        print(
            f"-z value at point of convergence: {ineq_constraint_1_3(x_s[-1], False)[0]:.4f}"
        )
        print(
            f"x + y + z value at point of convergence: {x_s[-1][0] + x_s[-1][1] + x_s[-1][2]:.6f}"
        )

        plot_iterations(
            "Objective function values of qp function",
            outer_obj_values,
            obj_values,
            "Outer objective values",
            "Objective values",

        )

        plot_feasible_regions_3d(x_s)

    def test_lp(self):
        x_s, obj_values, outer_x_s, outer_obj_values = self.minimizer.minimize(
            lp,
            self.starting_point_lp,
            [ineq_constraint_2_1, ineq_constraint_2_2, ineq_constraint_2_3, ineq_constraint_2_4],
            np.array([]),
            np.array([]),
            "wolfe",
            10e-12,
            10e-8,
            100,
            20,
            10e-10,
        )

        print(f"Point of convergence: {x_s[-1]}")
        print(
            f"Objective value at point of convergence: {lp(x_s[-1], False)[0]:.6f} (minimized over -x-y)"
        )
        print(
            f"-y -x +1 value at point of convergence: {ineq_constraint_2_1(x_s[-1], False)[0]:.6f}"
        )
        print(
            f"y - 1 value at point of convergence: {ineq_constraint_2_2(x_s[-1], False)[0]:.6f}"
        )
        print(
            f"x - 2 value at point of convergence: {ineq_constraint_2_3(x_s[-1], False)[0]:.6f}"
        )
        print(
            f"-y value at point of convergence: {ineq_constraint_2_4(x_s[-1], False)[0]:.6f}"
        )

        plot_feasible_set_2d(x_s)

        plot_iterations(
            "Objective function values of lp function",
            outer_obj_values,
            obj_values,
            "Outer objective values",
            "Objective values",
        )


if __name__ == "__main__":
    unittest.main()