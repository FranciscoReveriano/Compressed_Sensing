# Optimizer to solve undetermined linear system
import mosek
from mosek.fusion import *


def l1norm(X, y):
    n, m = X.shape

    M = Model("l1norm")
    w = M.variable(m)
    t = M.variable(n)

    M.objective(ObjectiveSense.Minimize, Expr.sum(t))

    res = Expr.sub(Expr.mul(X, w), y)
    M.constraint(Expr.sub(t, res), Domain.greaterThan(0))
    M.constraint(Expr.add(t, res), Domain.greaterThan(0))

    # M.setLogHandler(sys.stdout)
    M.solve()

    # Return the weights vector and the residuals
    w = w.level()
    return w, X.dot(w) - y