import numpy as np
import math

class InteriorPointMinimizer:
    T = 1
    MU = 10

    def minimize(
        self,
        func,
        x0,
        ineq_constraints,
        eq_constraints_mat,
        eq_constraints_rhs,
        step_len,
        obj_tol,
        param_tol,
        max_iter_inner,
        max_iter_outer,
        epsilon,
    ):

        x = x0
        f_x, g_x, h_x = func(x, True)
        f_x_phi, g_x_phi, h_x_phi = self.phi(ineq_constraints, x)
        t = self.T

        x_s = [x0]
        obj_values = [f_x]
        outer_x_s = [x0]
        outer_obj_values = [f_x]

        f_x = t * f_x + f_x_phi
        g_x = t * g_x + g_x_phi
        h_x = t * h_x + h_x_phi

        outer_iter = 0
        while outer_iter < max_iter_outer:

            if eq_constraints_mat.size:
                upper_block = np.concatenate([h_x, eq_constraints_mat.T], axis=1)
                size_zeros = (eq_constraints_mat.shape[0], eq_constraints_mat.shape[0])
                lower_block = np.concatenate(
                    [eq_constraints_mat, np.zeros(size_zeros)],
                    axis=1,
                )
                block_matrix = np.concatenate([upper_block, lower_block], axis=0)
            else:
                block_matrix = h_x
            eq_vec = np.concatenate([-g_x, np.zeros(block_matrix.shape[0] - len(g_x))])

            x_prev = x
            f_prev = f_x

            inner_iter = 0
            while inner_iter < max_iter_inner:

                if inner_iter != 0 and sum(abs(x - x_prev)) < param_tol:
                    break

                p = np.linalg.solve(block_matrix, eq_vec)[: len(x)]
                _lambda = np.matmul(p.transpose(), np.matmul(h_x, p)) ** 0.5
                if 0.5 * (_lambda**2) < obj_tol:
                    break

                if inner_iter != 0 and (f_prev - f_x < obj_tol):
                    break

                if step_len == "wolfe":
                    alpha = self.__wolfe(func, p, x)

                else:
                    alpha = step_len

                x_prev = x
                f_prev = f_x

                x = x + alpha * p
                f_x, g_x, h_x = func(x, True)
                f_x_phi, g_x_phi, h_x_phi = self.phi(ineq_constraints, x)

                x_s.append(x)
                obj_values.append(f_x)

                f_x = t * f_x + f_x_phi
                g_x = t * g_x + g_x_phi
                h_x = t * h_x + h_x_phi

                inner_iter += 1

            outer_x_s.append(x)
            outer_obj_values.append((f_x - f_x_phi) / t)

            if len(ineq_constraints) / t < epsilon:
                return x_s, obj_values, outer_x_s, outer_obj_values

            t = self.MU * t

            outer_iter += 1

        return x_s, obj_values, outer_x_s, outer_obj_values

    def __wolfe(self, f, p, x) -> int:

        alpha = 1
        wolfe_cons = 0.01
        backtrack_cons = 0.5

        while f(x + alpha * p, False)[0] > f(x, False)[
            0
        ] + wolfe_cons * alpha * np.matmul(f(x, False)[1].transpose(), p):
            alpha = alpha * backtrack_cons

        return alpha

    def phi(self, ineq_constraints, x):
        return_f = 0
        return_g = 0
        return_h = 0
        for func in ineq_constraints:
            f_x, g_x, h_x = func(x, True)
            return_f += math.log(-f_x)
            g = g_x / f_x
            return_g += g
            g_mesh = np.tile(
                g.reshape(g.shape[0], -1), (1, g.shape[0])
            ) * np.tile(g.reshape(g.shape[0], -1).T, (g.shape[0], 1))
            return_h += (h_x * f_x - g_mesh) / f_x**2

        return -return_f, -return_g, -return_h