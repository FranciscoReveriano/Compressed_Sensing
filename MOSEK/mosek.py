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


def l2norm(X, y):
    try:
        n, m = X.shape

        M = Model("l2norm")
        w = M.variable(m)
        t = M.variable()

        M.objective(ObjectiveSense.Minimize, t)

        res = Expr.sub(Expr.mul(X, w), y)
        M.constraint(Expr.vstack(t, res), Domain.inQCone())

        # M.setLogHandler(sys.stdout)
        M.solve()

        # Return the weights vector and the residuals
        w = w.level()
    finally:
        M.dispose()
    return w, X.dot(w) - y


# Implement the lasso part of the constraints, and return the bounding variable
def lassoVar(M, w, d):
    p = M.variable("p", d)
    plasso = M.variable("plasso")

    # plasso >= sum(p_i)
    M.constraint(Expr.sub(plasso, Expr.sum(p)), Domain.greaterThan(0.0))

    # p_i = |w_i|
    M.constraint(Expr.add(p, w), Domain.greaterThan(0.0))
    M.constraint(Expr.sub(p, w), Domain.greaterThan(0.0))

    return plasso


# Implement the ridge part of the constraints, and return the bounding variable
def ridgeVar(M, w):
    pridge = M.variable("pridge")
    M.constraint(Expr.vstack(0.5, pridge, w), Domain.inRotatedQCone())
    return pridge


# Regularized least-squares regression
def lseReg(X, y, lambda1=0, lambda2=0):
    n, d = len(X), len(X[0])
    M = Model("LSE-REG")

    # The regression coefficients
    w = M.variable("w", d)

    # The bound on the norm of the residual
    t = M.variable("t")
    r = Expr.sub(y, Expr.mul(X, w))
    # t \geq |r|_2^2
    M.constraint(Expr.vstack(0.5, t, r), Domain.inRotatedQCone())

    # The objective, add regularization terms as required
    objExpr = t.asExpr()
    if lambda1 != 0: objExpr = Expr.add(objExpr, Expr.mul(lambda1, lassoVar(M, w, d)))
    if lambda2 != 0: objExpr = Expr.add(objExpr, Expr.mul(lambda2, ridgeVar(M, w)))
    M.objective(ObjectiveSense.Minimize, objExpr)
    return M