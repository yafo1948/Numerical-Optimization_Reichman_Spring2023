import numpy as np

class LineSearchSolver:

    def __init__(self, method) -> None:
        self.method = method
        self.x_hist, self.f_x_hist = [], []

    def minimize_newton(self, f, x0, obj_tol, param_tol, max_iter):
        x = x0
        win = False

        f_x, g_x, h_x = f(x, True)
        print(f"i = 0, x0 = {x}, f(x0) = {f_x}")

        self.x_hist.append(x)
        self.f_x_hist.append(f_x)

        iter = 0
        while not win and iter < max_iter:
            f_x, g_x, h_x = f(x, True)
            direx = np.linalg.solve(h_x, -g_x)
            alpha = self.get_step(x, f, f_x, direx)
            x1 = x + direx * alpha
            f_x1, g_x, h_x = f(x1, True)

            iter += 1

            lamda = np.matmul(direx.T, np.matmul(h_x, direx)) ** 0.5
            if iter != 0 and (0.5 * (lamda ** 2) < obj_tol or np.linalg.norm(x1 - x) < param_tol):
                win = True
                print(f"iteration # {iter}, x{iter} = {x}, f(x{iter}) = {f_x}, win? {win}")
                return x, f_x, self.x_hist, self.f_x_hist, win

            x = x1
            f_x = f_x1
            # print(f'iteration # {iter}\n')
            # print(f'x: {x}\n')
            # print(f'f(x): {f_x}')
            # print('=' * 20, '\n\n')
            print(f"iteration # {iter}, x{iter} = {x}, f(x{iter}) = {f_x}, win? {win}")

            self.x_hist.append(x)
            self.f_x_hist.append(f_x)

        return x, f_x, self.x_hist, self.f_x_hist, win

    def minimize(self, f, x0, obj_tol, param_tol, max_iter):
        x = x0
        win = False

        f_x, g_x, h_x = f(x, False)

        print(f"i = 0, x0 = {x}, f(x0) = {f_x}")

        self.x_hist.append(x)
        self.f_x_hist.append(f_x)

        B = np.identity(len(x0))

        iter = 0
        while not win and iter < max_iter:
            f_x, g_x, h_x = f(x, False)
            if self.method == 'gradient descent':
                direx = -g_x
                alpha = self.get_step(x, f, f_x, direx)
                x1 = x + direx * alpha
                f_x1, g_x, h_x = f(x1, False)

            if self.method in ["BFGS", "sr1"]:
                f_x, g_x, h_x = f(x, False)

                direx = -np.linalg.solve(B, g_x)
                alpha = self.get_step(x, f, f_x, direx)
                x1 = x + direx * alpha
                f_x1, g_x1, h_x = f(x1, False)

                s = x1 - x
                y = g_x1 - g_x
                Bs = B @ s
                if self.method == "BFGS":
                    sBs = np.dot(s, Bs)
                    yTs = np.dot(y, s)
                    B = B - np.divide(np.outer(Bs, Bs), sBs) + np.divide(np.outer(y, y), yTs)
                else:
                    B = B + np.divide(np.outer(y - Bs, y - Bs), np.dot(y - Bs, s))

            iter += 1

            if iter != 0 and (abs(f_x1 - f_x) < obj_tol or np.linalg.norm(x1 - x) < param_tol):
                win = True
                print(f"iteration # {iter}, x{iter} = {x}, f(x{iter}) = {f_x}, win? {win}")
                return x, f_x, self.x_hist, self.f_x_hist, win

            x = x1
            f_x = f_x1
            # print(f'iteration # {iter}\n')
            # print(f'x: {x}\n')
            # print(f'f(x): {f_x}\n')
            # print('='*20, '\n\n')
            print(f"iteration # {iter}, x{iter} = {x}, f(x{iter}) = {f_x}, win? {win}")

            self.x_hist.append(x)
            self.f_x_hist.append(f_x)

        return x, f_x, self.x_hist, self.f_x_hist, win


    @staticmethod
    def get_step(x, f, f_x, direx) -> int:
        wolfe_cons = 0.01
        backtrack_cons = 0.5
        
        alpha = 1

        while f(x + alpha * direx, False)[0] > f_x + wolfe_cons * alpha * np.dot(-direx, direx):
            alpha = alpha * backtrack_cons

        return alpha
