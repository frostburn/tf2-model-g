def polynomial_order_4_centered_unoptimized(a0, b0, t, A, B, k4):
    a1 = a0**2*b0 + B*a0**2*k4/A + B*a0 + 2*A*a0*b0/k4 - a0*k4 + A**2*b0/k4**2
    b1 = -a0**2*b0 - B*a0**2*k4/A - B*a0 - 2*A*a0*b0/k4 - A**2*b0/k4**2
    a2 = a0*a1*b0 + 1/2*a0**2*b1 + B*a0*a1*k4/A + 1/2*B*a1 + A*a1*b0/k4 + A*a0*b1/k4 - 1/2*a1*k4 + 1/2*A**2*b1/k4**2
    b2 = -a0*a1*b0 - 1/2*a0**2*b1 - B*a0*a1*k4/A - 1/2*B*a1 - A*a1*b0/k4 - A*a0*b1/k4 - 1/2*A**2*b1/k4**2
    a3 = 1/3*a1**2*b0 + 2/3*a0*a2*b0 + 2/3*a0*a1*b1 + 1/3*a0**2*b2 + 1/3*B*a1**2*k4/A + 2/3*B*a0*a2*k4/A + 1/3*B*a2 + 2/3*A*a2*b0/k4 + 2/3*A*a1*b1/k4 + 2/3*A*a0*b2/k4 - 1/3*a2*k4 + 1/3*A**2*b2/k4**2
    b3 = -1/3*a1**2*b0 - 2/3*a0*a2*b0 - 2/3*a0*a1*b1 - 1/3*a0**2*b2 - 1/3*B*a1**2*k4/A - 2/3*B*a0*a2*k4/A - 1/3*B*a2 - 2/3*A*a2*b0/k4 - 2/3*A*a1*b1/k4 - 2/3*A*a0*b2/k4 - 1/3*A**2*b2/k4**2
    a4 = 1/2*a1*a2*b0 + 1/2*a0*a3*b0 + 1/4*a1**2*b1 + 1/2*a0*a2*b1 + 1/2*a0*a1*b2 + 1/4*a0**2*b3 + 1/2*B*a1*a2*k4/A + 1/2*B*a0*a3*k4/A + 1/4*B*a3 + 1/2*A*a3*b0/k4 + 1/2*A*a2*b1/k4 + 1/2*A*a1*b2/k4 + 1/2*A*a0*b3/k4 - 1/4*a3*k4 + 1/4*A**2*b3/k4**2
    b4 = -1/2*a1*a2*b0 - 1/2*a0*a3*b0 - 1/4*a1**2*b1 - 1/2*a0*a2*b1 - 1/2*a0*a1*b2 - 1/4*a0**2*b3 - 1/2*B*a1*a2*k4/A - 1/2*B*a0*a3*k4/A - 1/4*B*a3 - 1/2*A*a3*b0/k4 - 1/2*A*a2*b1/k4 - 1/2*A*a1*b2/k4 - 1/2*A*a0*b3/k4 - 1/4*A**2*b3/k4**2

    return (
        a0 + t * (a1 + t * (a2 + t * (a3 + t*a4))),
        b0 + t * (b1 + t * (b2 + t * (b3 + t*b4))),
    )


def polynomial_order_4_centered(a0, b0, t, A, B, k4):
    Ak4 = A/k4
    Ak4_2 = Ak4**2
    Bk4A = B/Ak4

    b1 = -a0*(a0*(b0 + Bk4A) + B + 2*Ak4*b0) - Ak4_2*b0
    a1 = -b1 - a0*k4
    b2 = -1/2*(a0*(2*a1*(b0 + Bk4A) + a0*b1 + 2*Ak4*b1) + a1*(B +  2*Ak4*b0) + Ak4_2*b1)
    a2 = -b2 - 1/2*a1*k4
    b3 = -1/3*(a0*(2*Ak4*b2 + a0*b2) + ((2*Bk4A+2*b0)*a0 + 2*Ak4*b0 + B)*a2 + ((Bk4A + b0)*a1 + 2*Ak4*b1 + 2*a0*b1)*a1 + Ak4_2*b2)
    a3 = -b3 - 1/3*a2*k4
    b4 = -1/4*(2*Ak4*a3*b0 + 2*Ak4*a2*b1 + (2*Bk4A*a2 + 2*a2*b0 + a1*b1 + 2*Ak4*b2)*a1 + (2*a3*(Bk4A + b0) + 2*a2*b1 + 2*a1*b2 + (2*Ak4 + a0)*b3)*a0 + B*a3 + Ak4_2*b3)
    a4 = -b4 - 1/4*a3*k4

    return (
        a0 + t * (a1 + t * (a2 + t * (a3 + t*a4))),
        b0 + t * (b1 + t * (b2 + t * (b3 + t*b4))),
    )


if __name__ == '__main__':
    from pylab import *
    N = 1000

    A = 3.4 + randn(N) * 0.1
    B = 13 + randn(N)
    k4 = 1.0 + randn(N) * 0.1

    x = randn()
    y = randn()

    dt = 0.1

    x_u, y_u = polynomial_order_4_centered_unoptimized(x, y, dt, A, B, k4)
    x_o, y_o = polynomial_order_4_centered(x, y, dt, A, B, k4)

    print("Unoptimized vs. optimized drift", abs(x_u - x_o).max(), abs(y_u - y_o).max())
