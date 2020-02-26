def steady_state(A, B, k2, k_2, k5):
    G0 = A*(k5 + k_2)/(k2*k5)
    X0 = A/k5
    Y0 = B*k5/A
    return G0, X0, Y0


def backward_euler(G, X, Y, dt, A, B, k2, k_2, k5, fixed_point_iterations=4):
    new_G = G
    new_X = X
    new_Y = Y
    for _ in range(fixed_point_iterations):
        gx_flow = k_2*new_X - k2*new_G
        xy_flow = new_X*new_X*new_Y - B*new_X
        v_G = A + gx_flow
        v_X = xy_flow - gx_flow - k5 * new_X
        v_Y = -xy_flow
        new_G = G + dt*v_G
        new_X = X + dt*v_X
        new_Y = Y + dt*v_Y
    return new_G, new_X, new_Y


def polynomial_order_4(a0, b0, c0, t, A, B, k2, k_2, k5):
    a1 = -a0*k2 + b0*k_2 + A
    b1 = b0**2*c0 - B*b0 + a0*k2 - b0*k5 - b0*k_2
    c1 = -b0**2*c0 + B*b0
    a2 = -1/2*a1*k2 + 1/2*b1*k_2
    b2 = b0*b1*c0 + 1/2*b0**2*c1 - 1/2*B*b1 + 1/2*a1*k2 - 1/2*b1*k5 - 1/2*b1*k_2
    c2 = -b0*b1*c0 - 1/2*b0**2*c1 + 1/2*B*b1
    a3 = -1/3*a2*k2 + 1/3*b2*k_2
    b3 = 1/3*b1**2*c0 + 2/3*b0*b2*c0 + 2/3*b0*b1*c1 + 1/3*b0**2*c2 - 1/3*B*b2 + 1/3*a2*k2 - 1/3*b2*k5 - 1/3*b2*k_2
    c3 = -1/3*b1**2*c0 - 2/3*b0*b2*c0 - 2/3*b0*b1*c1 - 1/3*b0**2*c2 + 1/3*B*b2
    a4 = -1/4*a3*k2 + 1/4*b3*k_2
    b4 = 1/2*b1*b2*c0 + 1/2*b0*b3*c0 + 1/4*b1**2*c1 + 1/2*b0*b2*c1 + 1/2*b0*b1*c2 + 1/4*b0**2*c3 - 1/4*B*b3 + 1/4*a3*k2 - 1/4*b3*k5 - 1/4*b3*k_2
    c4 = -1/2*b1*b2*c0 - 1/2*b0*b3*c0 - 1/4*b1**2*c1 - 1/2*b0*b2*c1 - 1/2*b0*b1*c2 - 1/4*b0**2*c3 + 1/4*B*b3

    return (
        a0 + t * (a1 + t * (a2 + t * (a3 + t*a4))),
        b0 + t * (b1 + t * (b2 + t * (b3 + t*b4))),
        c0 + t * (c1 + t * (c2 + t * (c3 + t*c4))),
    )


def polynomial_order_4_centered_unoptimized(a0, b0, c0, t, A, B, k2, k_2, k5):
    a1 = -a0*k2 + b0*k_2
    b1 = b0**2*c0 + B*b0**2*k5/A + B*b0 + a0*k2 + 2*A*b0*c0/k5 - b0*k5 - b0*k_2 + A**2*c0/k5**2
    c1 = -b0**2*c0 - B*b0**2*k5/A - B*b0 - 2*A*b0*c0/k5 - A**2*c0/k5**2
    a2 = -1/2*a1*k2 + 1/2*b1*k_2
    b2 = b0*b1*c0 + 1/2*b0**2*c1 + B*b0*b1*k5/A + 1/2*B*b1 + 1/2*a1*k2 + A*b1*c0/k5 + A*b0*c1/k5 - 1/2*b1*k5 - 1/2*b1*k_2 + 1/2*A**2*c1/k5**2
    c2 = -b0*b1*c0 - 1/2*b0**2*c1 - B*b0*b1*k5/A - 1/2*B*b1 - A*b1*c0/k5 - A*b0*c1/k5 - 1/2*A**2*c1/k5**2
    a3 = -1/3*a2*k2 + 1/3*b2*k_2
    b3 = 1/3*b1**2*c0 + 2/3*b0*b2*c0 + 2/3*b0*b1*c1 + 1/3*b0**2*c2 + 1/3*B*b1**2*k5/A + 2/3*B*b0*b2*k5/A + 1/3*B*b2 + 1/3*a2*k2 + 2/3*A*b2*c0/k5 + 2/3*A*b1*c1/k5 + 2/3*A*b0*c2/k5 - 1/3*b2*k5 - 1/3*b2*k_2 + 1/3*A**2*c2/k5**2
    c3 = -1/3*b1**2*c0 - 2/3*b0*b2*c0 - 2/3*b0*b1*c1 - 1/3*b0**2*c2 - 1/3*B*b1**2*k5/A - 2/3*B*b0*b2*k5/A - 1/3*B*b2 - 2/3*A*b2*c0/k5 - 2/3*A*b1*c1/k5 - 2/3*A*b0*c2/k5 - 1/3*A**2*c2/k5**2
    a4 = -1/4*a3*k2 + 1/4*b3*k_2
    b4 = 1/2*b1*b2*c0 + 1/2*b0*b3*c0 + 1/4*b1**2*c1 + 1/2*b0*b2*c1 + 1/2*b0*b1*c2 + 1/4*b0**2*c3 + 1/2*B*b1*b2*k5/A + 1/2*B*b0*b3*k5/A + 1/4*B*b3 + 1/4*a3*k2 + 1/2*A*b3*c0/k5 + 1/2*A*b2*c1/k5 + 1/2*A*b1*c2/k5 + 1/2*A*b0*c3/k5 - 1/4*b3*k5 - 1/4*b3*k_2 + 1/4*A**2*c3/k5**2
    c4 = -1/2*b1*b2*c0 - 1/2*b0*b3*c0 - 1/4*b1**2*c1 - 1/2*b0*b2*c1 - 1/2*b0*b1*c2 - 1/4*b0**2*c3 - 1/2*B*b1*b2*k5/A - 1/2*B*b0*b3*k5/A - 1/4*B*b3 - 1/2*A*b3*c0/k5 - 1/2*A*b2*c1/k5 - 1/2*A*b1*c2/k5 - 1/2*A*b0*c3/k5 - 1/4*A**2*c3/k5**2

    return (
        a0 + t * (a1 + t * (a2 + t * (a3 + t*a4))),
        b0 + t * (b1 + t * (b2 + t * (b3 + t*b4))),
        c0 + t * (c1 + t * (c2 + t * (c3 + t*c4))),
    )


def polynomial_order_4_centered(a0, b0, c0, t, A, B, k2, k_2, k5):
    """
    Integrate 0-dimensional Model G starting from a known position `t` time units ahead.
    The system has been "centered" so that the origin becomes a fixed point.
    The coefficients have been derived using a computer algebra system by developing the solution into a series with respect to t around G,X,Y == a0,b0,c0.
    """
    Ak5 = A/k5
    Ak5_2 = Ak5**2
    Bk5A = B*k5/A

    a1 = -a0*k2 + b0*k_2
    c1 = -b0*(b0*c0 + Bk5A*b0 + B + 2*Ak5*c0) - Ak5_2*c0
    b1 = -c1 + a0*k2 - b0*k5 - b0*k_2
    a2 = 1/2*(-a1*k2 + b1*k_2)
    c2 = -b0*(b1*(c0 + Bk5A) + c1*(1/2*b0 + Ak5)) - b1*(1/2*B + Ak5*c0) - 1/2*Ak5_2*c1
    b2 = -c2 + 1/2*a1*k2 - 1/2*b1*k5 - 1/2*b1*k_2
    a3 = 1/3*(-a2*k2 + b2*k_2)
    c3 = 1/3*(-b1*(b1*(c0 + Bk5A) + 2*Ak5*c1) - b0 * (2*b2*c0 + 2*b1*c1 + b0*c2 + 2*Bk5A*b2 + 2*Ak5*c2) - b2*(B + 2*Ak5*c0) - Ak5_2*c2)
    b3 = -c3 + 1/3*(a2*k2 - b2*k5 - b2*k_2)
    a4 = 1/4*(-a3*k2 + b3*k_2)
    c4 = 1/4*(-b1*(2*b2*(c0 + Bk5A) + b1*c1 + 2*Ak5*c2) - b0 *(2*b3*(c0 + Bk5A) + 2*b2*c1 + 2*b1*c2 + (b0 + 2*Ak5)*c3) - b3 * (B + 2*Ak5*c0) - 2*Ak5*b2*c1 - Ak5_2*c3)
    b4 = -c4 + 1/4*(a3*k2 - b3*k5 - b3*k_2)

    return (
        a0 + t * (a1 + t * (a2 + t * (a3 + t*a4))),
        b0 + t * (b1 + t * (b2 + t * (b3 + t*b4))),
        c0 + t * (c1 + t * (c2 + t * (c3 + t*c4))),
    )


if __name__ == '__main__':
    from pylab import *
    N = 1000

    A = 3.4 + randn(N) * 0.1
    B = 13 + randn(N)
    k2 = 1.0 + randn(N) * 0.1
    k_2 = 0.1 + randn(N) * 0.01
    k5 = 0.9 + randn(N) * 0.1

    G0, X0, Y0 = steady_state(A, B, k2, k_2, k5)

    assert (abs(array(backward_euler(G0, X0, Y0, 0.1, A, B, k2, k_2, k5)) - array([G0, X0, Y0])) < 1e-12).all()

    g = randn()
    x = randn()
    y = randn()

    G = G0 + g
    X = X0 + x
    Y = Y0 + y

    G_01, X_01, Y_01 = backward_euler(G, X, Y, 0.01, A, B, k2, k_2, k5)

    G_001, X_001, Y_001 = G, X, Y
    for _ in range(10):
        G_001, X_001, Y_001 = backward_euler(G_001, X_001, Y_001, 0.001, A, B, k2, k_2, k5)

    G_0001, X_0001, Y_0001 = G, X, Y
    for _ in range(100):
        G_0001, X_0001, Y_0001 = backward_euler(G_0001, X_0001, Y_0001, 0.0001, A, B, k2, k_2, k5)

    print("Backward Euler self-error scales as", abs(X_01 - X_0001).max(), abs(X_001 - X_0001).max())

    G_p01, X_p01, Y_p01 = polynomial_order_4(G, X, Y, 0.01, A, B, k2, k_2, k5)

    print("Polynomial order 4 integrator compares to Backward Euler as", abs(X_p01 - X_0001).max())

    G_p001, X_p001, Y_p001 = G, X, Y
    for _ in range(10):
        G_p001, X_p001, Y_p001 = polynomial_order_4(G_p001, X_p001, Y_p001, 0.001, A, B, k2, k_2, k5)

    G_p0001, X_p0001, Y_p0001 = G, X, Y
    for _ in range(100):
        G_p0001, X_p0001, Y_p0001 = polynomial_order_4(G_p0001, X_p0001, Y_p0001, 0.0001, A, B, k2, k_2, k5)

    print("Polynomial order 4 integrator self-error scales as", abs(X_p01 - X_p0001).max(), abs(X_p001 - X_p0001).max())

    G_pc01, X_pc01, Y_pc01 = polynomial_order_4_centered_unoptimized(g, x, y, 0.01, A, B, k2, k_2, k5) + array([G0, X0, Y0])

    print("Polynomial order 4 centered vs. plain", abs(X_p01 - X_pc01).max())

    g_u, x_u, y_u = g, x, y
    for _ in range(50):
        g_u, x_u, y_u = polynomial_order_4_centered_unoptimized(g_u, x_u, y_u, 0.02, A, B, k2, k_2, k5)

    g_o, x_o, y_o = g, x, y
    for _ in range(50):
        g_o, x_o, y_o = polynomial_order_4_centered(g_o, x_o, y_o, 0.02, A, B, k2, k_2, k5)

    print("Polynomial order 4 drift optimized vs. unoptimized", abs(x_u - x_o).max())
