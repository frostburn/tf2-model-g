from sage.all import *

X, Y, A, B, k4 = var("X, Y, A, B, k4")

X0 = A/k4
Y0 = B*k4/A

x = X + X0
y = Y + Y0

xy_flow = x*x*y - B*x
v_X = A + xy_flow - k4 * x
v_Y = -xy_flow

t = var("t")
coefs_a = var(",".join("a%d" % i for i in range(5)))
coefs_b = var(",".join("b%d" % i for i in range(5)))

poly_x = sum(a * t**i for i, a in enumerate(coefs_a))
poly_y = sum(b * t**i for i, b in enumerate(coefs_b))


v_poly_x = v_X.substitute(X==poly_x, Y==poly_y).coefficients(t, sparse=False)
v_poly_y = v_Y.substitute(X==poly_x, Y==poly_y).coefficients(t, sparse=False)
d_poly_x = diff(poly_x, t).coefficients(t, sparse=False)
d_poly_y = diff(poly_y, t).coefficients(t, sparse=False)

print("def polynomial_order_4_centered(a0, b0, t, A, B, k4):")
for i in range(4):
    sol = solve([d_poly_x[i] == v_poly_x[i], d_poly_y[i] == v_poly_y[i]], coefs_a[i+1], coefs_b[i+1])
    for s in sol[0]:
        print("    " + str(s.expand()).replace("==", "=").replace("^", "**"))
print("""   return (
        a0 + t * (a1 + t * (a2 + t * (a3 + t*a4))),
        b0 + t * (b1 + t * (b2 + t * (b3 + t*b4))),
    )""")
