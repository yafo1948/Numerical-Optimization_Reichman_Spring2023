import numpy as np

class LineSearchSolver:
    wolfe_cons = 0.01
    backtrack_cons = 0.5

    def __init__(self, method) -> None:
        self.method = method

    def minimize(self, f, x0, step_len, obj_tol, param_tol, max_iter):

        x = x0
        f_x, g_x, h_x = f(x, True)

        print(f"i = 0, x0 = {x}, f(x0) = {f_x}")

        x_prev = x
        f_prev = f_x

        x_hist = [x0]
        f_x_hist = [f_x]
        
        if self.method in ["BFGS", "sr1"]:
            B = np.identity(len(x0))

        iter = 0
        while iter < max_iter:

            if iter != 0 and np.linalg.norm(x - x_prev) < param_tol: #sum(abs(x - x_prev))
                return x, f_x, x_hist, f_x_hist, True

            if self.method == "Newton":
                direx = np.linalg.solve(h_x, -g_x)
                _lambda = np.matmul(direx.transpose(), np.matmul(h_x, direx)) ** 0.5
                if 0.5 * (_lambda ** 2) < obj_tol:
                    return x, f_x, x_hist, f_x_hist, True

            if self.method in ["BFGS", "sr1"]:
                f_prev, g_x, h_x = f(x_prev, False)
                
                direx = -np.linalg.solve(B, g_x)
                step_size = self.get_step(f, direx, x_prev)
                x = x_prev + direx * step_size
                f_x, g_next, h_x = f(x, False)

                s = x - x_prev
                y = g_next - g_x
                Bs = B @ s
                if self.method == "BFGS":
                    sBs = np.dot(s, Bs)
                    yTs = np.dot(y, s)
                    B = B - np.divide(np.outer(Bs, Bs), sBs) + np.divide(np.outer(y, y), yTs)
                else: 
                    B = B + np.divide(np.outer(y - Bs, y - Bs), np.dot(y - Bs, s))
                
            
            else:
                direx = -g_x

            if iter != 0 and (f_prev - f_x < obj_tol):
                return x, f_x, x_hist, f_x_hist, True

            if step_len == "wolfe":
                alpha = self.get_step(f, direx, x)

            else:
                alpha = step_len

            x_prev = x
            f_prev = f_x

            x = x + alpha * direx
            
            if self.method == "Newton":
                f_x, g_x, h_x = f(x, True)
            else:
                f_x, g_x = f(x, False)

            print(f"i = {iter + 1}, x{iter + 1} = {x}, f(x{iter + 1}) = {f_x}")

            x_hist.append(x)
            f_x_hist.append(f_x)

            iter += 1

        return x, f_x, x_hist, f_x_hist, False

    def get_step(self, f, direx, x) -> int:
        alpha = 1

        while f(x + alpha * direx, False)[0] > f(x, False)[0] + self.wolfe_cons * alpha * np.matmul(f(x, False)[1].transpose(), direx):
            alpha = alpha * self.backtrack_cons

        return alpha
