from sage.all import *

G, X, Y, A, B, k2, k_2, k5 = var("G, X, Y, A, B, k2, k_2, k5")

G0 = A*(k5 + k_2)/(k2*k5)
X0 = A/k5
Y0 = B*k5/A

g = G + G0
x = X + X0
y = Y + Y0

gx_flow = k_2*x - k2*g
xy_flow = x*x*y - B*x
v_G = A + gx_flow
v_X = -gx_flow + xy_flow - k5 * x
v_Y = -xy_flow

t = var("t")
coefs_a = var(",".join("a%d" % i for i in range(5)))
coefs_b = var(",".join("b%d" % i for i in range(5)))
coefs_c = var(",".join("c%d" % i for i in range(5)))

poly_g = sum(a * t**i for i, a in enumerate(coefs_a))
poly_x = sum(b * t**i for i, b in enumerate(coefs_b))
poly_y = sum(c * t**i for i, c in enumerate(coefs_c))


v_poly_g = v_G.substitute(G==poly_g, X==poly_x, Y==poly_y).coefficients(t, sparse=False)
v_poly_x = v_X.substitute(G==poly_g, X==poly_x, Y==poly_y).coefficients(t, sparse=False)
v_poly_y = v_Y.substitute(G==poly_g, X==poly_x, Y==poly_y).coefficients(t, sparse=False)
d_poly_g = diff(poly_g, t).coefficients(t, sparse=False)
d_poly_x = diff(poly_x, t).coefficients(t, sparse=False)
d_poly_y = diff(poly_y, t).coefficients(t, sparse=False)

print("def polynomial_order_4_centered(a0, b0, c0, t, A, B, k2, k_2, k5):")
for i in range(4):
    sol = solve(
        [d_poly_g[i] == v_poly_g[i], d_poly_x[i] == v_poly_x[i], d_poly_y[i] == v_poly_y[i]],
        coefs_a[i+1], coefs_b[i+1], coefs_c[i+1])
    for s in sol[0]:
        print("    " + str(s.expand()).replace("==", "=").replace("^", "**"))

print("""   return (
        a0 + t * (a1 + t * (a2 + t * (a3 + t*a4))),
        b0 + t * (b1 + t * (b2 + t * (b3 + t*b4))),
        c0 + t * (c1 + t * (c2 + t * (c3 + t*c4))),
    )""")