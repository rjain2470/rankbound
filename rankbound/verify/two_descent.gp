\\ The 2-isogeny descent of rankbound, for curves E/Q with a rational point of order 2: an unconditional bound on
\\ rank E(Q), and a lower bound from points.  Loaded by the verifier (tdverify) and by the prover (tdprove); one
\\ function per step: the model, the local images W_v, the Selmer groups, the bound, the characters of the points.
\\
\\ E is the reduced minimal model and T(X) = X^3 + b2 X^2 + 8 b4 X + 16 b6: (x, y) -> (4x, 8y + 4 a1 x + 4 a3) maps E
\\ isomorphically onto Y^2 = T(X).  T is monic and integral, so its rational roots are integers; t = dim E(Q)[2] is 1
\\ (one root) or 2 (three).  For a root e, X = X' + e gives Y^2 = X'(X'^2 + a X' + b), and the 2-isogenous curve is
\\ Y^2 = X(X^2 + a' X + b'), a' = -2a, b' = a^2 - 4b [Cremona, Algorithms for Modular Elliptic Curves, 2nd ed., 3.5;
\\ Silverman, The Arithmetic of Elliptic Curves, X.4.9].
\\
\\ For (A, B) = (a, b) or (a', b'), Sel(A, B) is the set of d in Q(S, 2) (S: the primes of 2 disc E; Q(S, 2): the
\\ classes of +-prod_{p in S} p^e_p in Q*/Q*^2) such that  w^2 = d u^4 + A u^2 v^2 + (B/d) v^4  has a point with
\\ (u, v) != (0, 0) over R and over Q_p for every p in S.  It is the image of the connecting map of the isogeny with
\\ kernel (0, 0) on the other curve: Sel(a, b) contains E(Q)/psi(E'(Q)) (the map (X, Y) -> X, (0, 0) -> b) and
\\ Sel(a', b') contains E'(Q)/phi(E(Q)).  Outside S, p is odd and divides neither B nor A^2 - 4B, and the local image
\\ is the unit classes, which Q(S, 2) builds in.  Each local condition is "the class of d at v lies in W_v", a
\\ subgroup of Q_v*/Q_v*^2, so Sel is the kernel of the F_2-linear map Q(S, 2) -> prod_v (Q_v*/Q_v*^2) / W_v.
\\
\\ The bound of the 2-descent.  With s = dim Sel(a, b), s' = dim Sel(a', b'), the exact sequence
\\   0 -> E'(Q)[psi] / phi(E(Q)[2]) -> Sel^phi(E) -> Sel_2(E) -> Sel^psi(E')
\\ (the left term has dimension 1 if t = 1, 0 if t = 2) gives dim Sel_2(E) - t <= s + s' - 2 = B_e.  Then
\\ rank E(Q) <= r_inf <= dim Sel_2(E) - t, r_inf the Z_2-corank of the 2^infinity-Selmer group, and (-1)^r_inf = w,
\\ the root number (the 2-parity theorem: Monsky 1996 for Sel_2; T. and V. Dokchitser, Crelle 2011).  So rank E(Q)
\\ <= S, the largest integer <= B = min_e B_e with (-1)^S = w.  No GRH, no BSD.
\\
\\ The second step, the isogeny pairings Theta_1 on Sel(a, b) and Theta_1' on Sel(a', b') (below), lowers B_e by their
\\ ranks: B_e = s + s' - 2 - rank Theta_1 - rank Theta_1'.  The third step, where that is still above what is known, is
\\ the Cassels-Tate pairing on Sel_2(E) and on Sel_2(E') (below): B_e is the least of the bounds.

\\ ---------- the model

\\ [E, T, the integer roots of T, sorted]: E the reduced minimal model of the a-invariants ai, given the primes P of
\\ its discriminant (the prime table is cleared first, so that long loops stay fast)
tdmodel(ai, P) =
{
  my(E, T);
  removeprimes(addprimes()); addprimes(P);
  E = ellminimalmodel(ellinit(ai));
  T = x^3 + E.b2*x^2 + 8*E.b4*x + 16*E.b6;
  [E, T, vecsort(Vec(nfroots(, T)))];
}

\\ [a, b] with T(X + e) = X (X^2 + a X + b), e a root of T
tdab(T, e) =
{
  my(V = Vec(subst(T, x, x + e)));
  if(V[4], error("tdab: ", e, " is not a root of T"));
  [V[2], V[3]];
}

\\ the 2-isogenous curve: [a', b'] = [-2a, a^2 - 4b]
tdisog(ab) = [-2 * ab[1], ab[1]^2 - 4 * ab[2]];

\\ ---------- square classes

\\ the least quadratic non-residue mod the odd prime p
tdeps(p) = my(e = 2); while(kronecker(e, p) != -1, e++); e;

\\ the F_2-basis of Q_v*/Q_v*^2 used here: -1 for R (v = 0); -1, 5, 2 for Q_2; eps, p for odd p
tdbasis(p) = if(p == 0, [-1], p == 2, [-1, 5, 2], [tdeps(p), p]);

\\ the coordinates in that basis of the class of the rational d != 0 at v: the sign at R; at p = 2 the unit
\\ u = d / 2^v(d) is (-1)^i 5^k mod 8 (1 -> 00, 3 -> 11, 5 -> 01, 7 -> 10); at odd p, [u non-residue, v(d) odd]
tdloc(d, p) =
{
  my(v, u);
  if(p == 0, return([d < 0]));
  v = valuation(d, p); u = d / p^v;
  if(p == 2, u = lift(Mod(u, 8)); return([u % 4 == 3, u == 3 || u == 5, v % 2]));
  [kronecker(lift(Mod(u, p)), p) == -1, v % 2];
}

\\ the class with coordinates c, as an integer
tdrep(c, p) = my(bs = tdbasis(p)); prod(i = 1, #bs, bs[i]^c[i]);

\\ ---------- local solubility of y^2 = g(u, v), g a binary quartic over Z_p
\\ Every point of P^1(Q_p) is (s : 1) with s in Z_p or (1 : p s) with s in Z_p, so y^2 = g(u, v) has a Q_p-point iff
\\ g(s, 1) or g(1, p s) is a square in Q_p for some s in Z_p.  That is decided by refining residue classes of s, as
\\ in Merriman-Siksek-Smart, Explicit 4-descents on an elliptic curve, Acta Arith. 77 (1996), section 6, and Cremona,
\\ op. cit., 3.6 (Lemmas 6 and 7), here with the class s = s0 + p s' written as a new polynomial in s'.  Since g
\\ is separable, every branch of the refinement ends: in a small class about a point with g != 0 the value has a fixed
\\ valuation and unit class, and about a simple root the polynomial becomes linear in s' with a unit slope.

TDENUM = 64;    \\ below this prime, tdsquare tries every residue

\\ 1 if H(s) is a nonzero square mod p for some s in F_p (p odd, H mod p != 0).  For p >= 17 and deg H <= 4, write
\\ H = c h^2 q mod p, q squarefree: if deg q >= 1, Weil's bound |sum_s (c q(s) / p)| <= (deg q - 1) sqrt(p) (Lidl and
\\ Niederreiter, Finite Fields, Thm. 5.41) leaves more values c q(s) that are nonzero squares than roots of h; if q is
\\ constant, the answer is whether c is a square, as h (deg <= 2) has a non-root.
tdsquare(H, p) =
{
  my(h = H * Mod(1, p), F);
  if(p < TDENUM,
    for(s = 0, p - 1, my(z = lift(subst(h, x, s))); if(z && kronecker(z, p) == 1, return(1)));
    return(0));
  if(poldegree(h) <= 0, return(kronecker(lift(polcoef(h, 0)), p) == 1));
  F = factor(h);
  for(i = 1, #F~, if(F[i, 2] % 2, return(1)));
  kronecker(lift(pollead(h)), p) == 1;
}

\\ the roots in [0, p) of H mod p (H mod p != 0)
tdroots(H, p) = apply(lift, Vec(polrootsmod(H, p)));

\\ 1 if H(s) is a square in Q_p (0 included) for some s in Z_p; H in Z[x], nonzero, separable over Q_p.  With c the
\\ valuation of the content: c >= 2: H / p^2; c = 1: a square needs H(s)/p = 0 mod p; c = 0, p odd: a unit square
\\ mod p lifts (Hensel), else H(s) = 0 mod p.
tdzp(H, p) =
{
  my(c = valuation(content(H), p), R);
  if(c >= 2, return(tdzp(H / p^(2 * (c \ 2)), p)));
  if(c == 1, R = tdroots(H / p, p),
    if(p == 2, return(tdz2(H)));
    if(tdsquare(H, p), return(1));
    R = tdroots(H, p));
  for(i = 1, #R, if(tdzp(subst(H, x, R[i] + p * x), p), return(1)));
  0;
}

\\ tdzp at p = 2 with H of content 1: on s = s0 + 8 s', H(s) = H(s0) mod 8, and an odd number is a square in Z_2 iff
\\ it is 1 mod 8; the classes with H(s0) even are refined
tdz2(H) =
{
  for(s0 = 0, 7, my(h = subst(H, x, s0));
    if(h == 0, return(1));
    if(h % 2, if(h % 8 == 1, return(1)), if(tdzp(subst(H, x, s0 + 8 * x), 2), return(1))));
  0;
}

\\ 1 if y^2 = g(u, v) has a Q_p-point with (u, v) != (0, 0), g = [g4, g3, g2, g1, g0] integral, separable
tdquartic(g, p) =
{
  tdzp(Pol(g), p) || tdzp(subst(Polrev(g), x, p * x), p);
}

\\ 1 if w^2 = d u^4 + A u^2 v^2 + (B/d) v^4 has a point over Q_v (p = 0: over R) with (u, v) != (0, 0).  Over R:
\\ with d < 0 and z = d u^2 / v^2 <= 0 it needs z^2 + A z + B <= 0, i.e. B < 0 or A > 0 with A^2 >= 4B.  Over Q_p:
\\ the quartic times d^2, d^3 u^4 + A d^2 u^2 v^2 + B d v^4, which is integral.
tdqp(A, B, d, p) =
{
  if(p == 0, return(d > 0 || B < 0 || (A > 0 && A^2 >= 4 * B)));
  tdquartic([d^3, 0, A * d^2, 0, B * d], p);
}

\\ ---------- the local images and the Selmer groups

\\ W_v for (A, B): the coordinate vectors of the classes d of Q_v*/Q_v*^2 at which the quartic is soluble; it is a
\\ subgroup (checked)
tdW(A, B, p) =
{
  my(n = #tdbasis(p), W = List());
  forvec(c = vector(n, i, [0, 1]), if(tdqp(A, B, tdrep(c, p), p), listput(W, c)));
  W = Set(Vec(W));
  foreach(W, u, foreach(W, v, if(!setsearch(W, (u + v) % 2), error("tdW: W_", p, " is not a group"))));
  W;
}

\\ the local images [[v, W_v], ...] for v = 0 (R) and the primes of S
tdimages(A, B, S) = vector(#S + 1, i, my(p = if(i == 1, 0, S[i - 1])); [p, tdW(A, B, p)]);

\\ the F_2-basis of Q(S, 2): -1 and the primes of S
tdgens(S) = concat([-1], S);

\\ the coordinates in tdgens(S) of the class of the integer d in Q*/Q*^2, or the integer 0 if a prime outside S
\\ divides d (or d = 0)
tdclass(d, S) =
{
  my(c = vector(#S + 1), m = abs(d));
  if(d == 0, return(0));
  c[1] = d < 0;
  for(i = 1, #S, my(v = valuation(m, S[i])); c[i + 1] = v % 2; m /= S[i]^v);
  if(m != 1, 0, c);
}

\\ Sel(A, B) from its local images Ws: a basis, as signed squarefree integers.  Each W_v gives the linear forms
\\ phi on Q_v*/Q_v*^2 vanishing on W_v (the kernel of the matrix of W_v), and d is in Sel iff phi(loc_v(d)) = 0
\\ for all of them.
tdselmer(S, Ws) =
{
  my(G = tdgens(S), cols = List(), M, K);
  foreach(Ws, pw, my(p = pw[1], W = pw[2], L, F);
    L = matrix(#G, #tdbasis(p), i, j, tdloc(G[i], p)[j]);
    F = lift(matker(Mod(matrix(#W, #tdbasis(p), i, j, W[i][j]), 2)));
    for(k = 1, #F, listput(cols, L * F[, k])));
  K = if(#cols, lift(matker(Mod(Mat(Vec(cols))~, 2))), matid(#G));
  vector(#K, k, prod(i = 1, #G, G[i]^K[i, k]));
}

\\ the local duality check at each place: W_v(a, b) and W_v(a', b') are orthogonal complements under the Hilbert
\\ symbol (Tate local duality for the isogeny and its dual, whose cup product with the Weil pairing is the Hilbert
\\ symbol on Q_v*/Q_v*^2): the dimensions add up to dim Q_v*/Q_v*^2 and (d, d')_v = 1 on W_v x W'_v
tddual(Ws, Ws2) =
{
  for(i = 1, #Ws, my(p = Ws[i][1], W = Ws[i][2], W2 = Ws2[i][2]);
    if(#W * #W2 != 2^#tdbasis(p), return(0));
    foreach(W, u, foreach(W2, v, if(hilbert(tdrep(u, p), tdrep(v, p), p) != 1, return(0)))));
  1;
}

\\ the descent by the 2-isogeny with kernel (e, 0): [e, a, b, Sel(a, b), Sel(a', b'), s, s', B_e = s + s' - 2,
\\ the duality check, whether b and b' lie in the Selmer groups]
tdisogeny(T, e, S) =
{
  my(ab = tdab(T, e), ab2 = tdisog(ab), W = tdimages(ab[1], ab[2], S), W2 = tdimages(ab2[1], ab2[2], S), s1, s2);
  s1 = tdselmer(S, W); s2 = tdselmer(S, W2);
  [e, ab[1], ab[2], s1, s2, #s1, #s2, #s1 + #s2 - 2, tddual(W, W2),
   tdinspan(ab[2], s1, S) && tdinspan(ab2[2], s2, S)];
}

\\ 1 if the class of the integer d is in the span of the basis V of a subgroup of Q(S, 2)
tdinspan(d, V, S) =
{
  my(c = tdclass(d, S), M);
  if(type(c) == "t_INT", return(0));
  if(#V == 0, return(c == 0 * c));
  M = Mat(apply(v -> tdclass(v, S)~, V));
  matrank(Mod(matconcat([M, c~]), 2)) == matrank(Mod(M, 2));
}

\\ 1 if the integers V are independent in Q(S, 2) and span the subgroup with basis V0
tdsamespan(V, V0, S) =
{
  my(C = apply(v -> tdclass(v, S), V), M, M0);
  if(#V != #V0 || #select(c -> type(c) == "t_INT", C), return(0));
  if(#V == 0, return(1));
  M = Mat(apply(c -> c~, C)); M0 = Mat(apply(v -> tdclass(v, S)~, V0));
  matrank(Mod(M, 2)) == #V && matrank(Mod(matconcat([M, M0]), 2)) == #V;
}

\\ ---------- the isogeny pairings Theta_1 (the second step)
\\ T. Fisher, Higher descents on an elliptic curve with a rational 2-torsion point, Math. Comp. 86 (2017) [F17].
\\ The covering curve of xi in Sel(A, B) is C_xi: r^2 = xi s^4 + A s^2 t^2 + (B/xi) t^4, which maps to the conic
\\ Q_xi: W^2 = xi X^2 + A X Y + (B/xi) Y^2 by (X : Y : W) = (s^2 : t^2 : r).  Q_xi is everywhere locally soluble, so it
\\ has a rational point P0; let L_xi be the tangent line to Q_xi at P0.  Then L_xi(s^2, t^2, r) is, up to squares, the
\\ pushout function f2 = f(l, m)/m^2 of [F17, sec. 5] (parametrise Q_xi with P0 at (l : m) = (1 : 0); then m^2 is
\\ L_xi, and X / L_xi = L_xi X / L_xi^2), and by [F17, sec. 3] the pairing on Sel(A, B) is
\\   Theta_1(xi, eta) = sum over all places v of (L_xi(P_v), eta)_v,   P_v in C_xi(Q_v), L_xi(P_v) != 0,
\\ ( , )_v the quadratic Hilbert symbol, each term independent of P_v.  By [F17, Thm. 2.1] (m = 1) Theta_1 is
\\ alternating, and its kernel is S_2, the image of the next Selmer group, which contains the image of the
\\ Mordell-Weil group.  So with Theta_1 on Sel(a, b) and Theta_1' on Sel(a', b'), and rank E(Q) = dim E(Q)/psi E'(Q)
\\ + dim E'(Q)/phi E(Q) - 2 [F17, sec. 2, with E(Q)[phi] and E'(Q)[phihat] of order 2],
\\   rank E(Q) <= dim S_2 + dim S_2' - 2 = s + s' - 2 - rank Theta_1 - rank Theta_1'.
\\ This also bounds r_inf: S_2 + S_2' - 2 is at least the 4-descent bound dim S_3 + dim S_4' - 2 [F17, Lemma 2.2],
\\ which bounds r_inf, so the parity step of tdbound stays valid.  The value does not depend on P0: two tangent lines
\\ differ by a constant c times a square on Q_xi, and sum_v (c, eta)_v = 0 (the product formula).
\\ Only the places V = {R} u S u {3, 5, 7} contribute.  For p outside V: xi, B/xi, A^2 - 4B and eta are p-units (they
\\ are supported on S, checked), so C_xi and Q_xi have good reduction at p; L_xi is primitive, so it vanishes at <= 2
\\ points of Q_xi mod p and <= 4 points of C_xi mod p, while #C_xi(F_p) >= p + 1 - 2 sqrt(p) > 4 for p >= 11; a smooth
\\ point with L_xi a unit lifts (Hensel), and (unit, unit)_p = 1 for odd p.

\\ the places that can contribute: 0 (R), the primes of S (2 among them), and 3, 5, 7
tdplaces(S) = concat([0], setunion(S, [3, 5, 7]));

\\ q(s, t) = xi s^4 + A s^2 t^2 + (B/xi) t^4
tdq(xi, A, B, s, t) = xi * s^4 + A * s^2 * t^2 + B / xi * t^4;

\\ 1 if the integer n != 0 has all its prime factors in S
tdsunit(n, S) = my(m = abs(n)); if(!m, return(0)); foreach(S, p, m /= p^valuation(m, p)); m == 1;

\\ a rational point [X0, Y0, W0] (primitive integers) of Q_xi: qfsolve on 2 xi times its form, whose determinant
\\ -2 xi^3 (4B - A^2) is factored over S
tdconic(xi, A, B, S) =
{
  my(G = [2 * xi^2, A * xi, 0; A * xi, 2 * B, 0; 0, 0, -2 * xi], d = abs(matdet(G)), Ps, P);
  if(!tdsunit(d, S), error("tdconic: det not supported on S"));
  Ps = select(p -> d % p == 0, S);
  P = qfsolve([G, matrix(#Ps, 2, i, j, if(j == 1, Ps[i], valuation(d, Ps[i])))]);
  if(type(P) != "t_COL", error("tdconic: the conic of ", xi, " has no rational point (", P, ")"));
  P = P / content(P); P~;
}

\\ the tangent line [alpha, beta, gamma] (primitive integers) to Q_xi at the rational point P0 = [X0, Y0, W0]:
\\ alpha X + beta Y + gamma W = 0, from the gradient of xi X^2 + A X Y + (B/xi) Y^2 - W^2
tdtangent(xi, A, B, P0) =
{
  my(l = [2 * xi * P0[1] + A * P0[2], A * P0[1] + 2 * B / xi * P0[2], -2 * P0[3]]);
  l = l * denominator(l); l / content(l);
}

\\ the class in Q_v*/Q_v*^2 of L(s^2, t^2, r) at the point (s, t, r) of C_xi over Q_v (r^2 = q(s, t)), as a rational
\\ number; 0 if q(s, t) is not a nonzero square in Q_v or the class is not fixed by 100 p-adic digits (or L = 0).
\\ Over R, r > 0 and the sign is exact; over Q_p, r is the p-adic square root of q (either sign gives the same
\\ pairing).
tdlclass(xi, A, B, L, s, t, p) =
{
  my(q = tdq(xi, A, B, s, t), u, r, l, v, k);
  if(q == 0, return(0));
  u = L[1] * s^2 + L[2] * t^2;
  if(p == 0,                    \\ the sign of u + L[3] sqrt(q), exactly
    if(q < 0, return(0));
    if(u == 0 || L[3] == 0 || sign(u) == sign(L[3]), return(if(u, sign(u), sign(L[3]))));
    k = sign(u^2 - L[3]^2 * q);
    return(if(k > 0, sign(u), k < 0, sign(L[3]), 0)));
  v = valuation(q, p);
  if(v % 2, return(0));
  if(p == 2, if((q / 2^v) % 8 != 1, return(0)), if(kronecker(lift(Mod(q / p^v, p)), p) != 1, return(0)));
  r = sqrt(q + O(p^(v + 100)));
  l = u + L[3] * r;
  if(l == 0, return(0));
  if(type(l) != "t_PADIC", return(l));          \\ L[3] = 0: l = u exactly
  v = valuation(l, p); k = padicprec(l, p) - v;
  if(k < if(p == 2, 3, 1), return(0));
  p^v * lift(l / p^v);
}

\\ ---------- the prover's local points

\\ a residue s mod p (odd) with H(s) a nonzero square mod p, or [] if there is none (tdsquare)
tdsqwit(H, p) =
{
  my(h = H * Mod(1, p));
  if(p < TDENUM, for(s = 0, p - 1, my(z = lift(subst(h, x, s))); if(z && kronecker(z, p) == 1, return(s))); return([]));
  if(!tdsquare(H, p), return([]));
  for(k = 1, 10^4, my(s = random(p), z = lift(subst(h, x, s))); if(z && kronecker(z, p) == 1, return(s)));
  error("tdsqwit: no residue found mod ", p);
}

\\ an integer s with H(s) a nonzero square in Q_p, or [] if there is none: tdzp with witnesses (all classes with a
\\ unit square are tried before the classes of the roots, so that a branch about a root ends)
tdwit(H, p) =
{
  my(c = valuation(content(H), p), R, q = p, w);
  if(c >= 2, return(tdwit(H / p^(2 * (c \ 2)), p)));
  if(c == 1, R = tdroots(H / p, p),
    if(p == 2,
      for(s = 0, 7, if(subst(H, x, s) % 8 == 1, return(s)));
      R = select(s -> subst(H, x, s) % 2 == 0, [0..7]); q = 8,
      w = tdsqwit(H, p); if(type(w) != "t_VEC", return(w));
      R = tdroots(H, p)));
  for(i = 1, #R, w = tdwit(subst(H, x, R[i] + q * x), p); if(type(w) != "t_VEC", return(R[i] + q * w)));
  [];
}

\\ a real point [s, t] of C_xi with L(s^2, t^2, r) != 0
tdreal(xi, A, B, L) =
{
  my(X1, c);
  for(n = 1, 40, for(s = 0, n, my(t = n - s); if(gcd(s, t) == 1 && tdlclass(xi, A, B, L, s, t, 0), return([s, t]))));
  \\ xi < 0 and B/xi < 0: q > 0 for s^2/t^2 between the positive roots of xi X^2 + A X + B/xi
  X1 = -A / (2 * xi);
  if(X1 > 0, localprec(60 + 2 * #Str(numerator(X1)) + 2 * #Str(denominator(X1)));
    for(k = 1, 400, my(z = bestappr(sqrt(X1 * 1.), 2^k)); c = [numerator(z), denominator(z)];
      if(tdlclass(xi, A, B, L, c[1], c[2], 0), return(c))));
  error("tdreal: no real point found for ", xi);
}

\\ a point [s, t] of C_xi over Q_p (p = 0: R) with L(s^2, t^2, r) of known class: a witness of tdwit on the integral
\\ quartic xi^2 q, moved within its residue class (which keeps the class of q) until L's class is fixed
tdlocal(xi, A, B, L, p) =
{
  my(g = [xi^3, 0, A * xi^2, 0, B * xi], w, s, t, K);
  if(p == 0, return(tdreal(xi, A, B, L)));
  w = tdwit(Pol(g), p);
  if(type(w) != "t_VEC", s = w; t = 1,
    w = tdwit(subst(Polrev(g), x, p * x), p);
    if(type(w) == "t_VEC", error("tdlocal: ", xi, " is not in the Selmer group at ", p));
    s = 1; t = p * w);
  K = valuation(xi^2 * tdq(xi, A, B, s, t), p) + 4;
  for(j = 0, 200, my(s1 = s + if(t == 1, j * p^K, 0), t1 = t + if(t == 1, 0, j * p^K));
    if(tdlclass(xi, A, B, L, s1, t1, p), return([s1, t1])));
  error("tdlocal: no point with L != 0 for ", xi, " at ", p);
}

\\ ---------- the pairing matrix

\\ the matrix of Theta_1 on the basis Xi of Sel(A, B) (rows: xi; columns: eta), from the tangent lines Ls and the local
\\ points Pts[i][k] at the places V[k]; 0 if a local point is not one (tdlclass)
tdtheta(A, B, V, Xi, Ls, Pts) =
{
  my(n = #Xi, M = matrix(n, n), c);
  for(i = 1, n, for(k = 1, #V,
    c = tdlclass(Xi[i], A, B, Ls[i], Pts[i][k][1], Pts[i][k][2], V[k]);
    if(!c, return(0));
    for(j = 1, n, M[i, j] += hilbert(c, Xi[j], V[k]) == -1)));
  M % 2;
}

\\ 1 if M is alternating over F_2
tdalt(M) = M == M~ && vecsum(vector(#M, i, M[i, i])) == 0;

\\ the pairing on Sel(A, B) with basis Xi: [conic points, local points, the matrix, its rank]; checks (a failure is
\\ a bug) that it is alternating and that B, the image of the 2-torsion point, is in its kernel
tdpairing(A, B, S, Xi) =
{
  my(V = tdplaces(S), C = apply(xi -> tdconic(xi, A, B, S), Xi), Ls, Pts, M, k);
  Ls = vector(#Xi, i, tdtangent(Xi[i], A, B, C[i]));
  Pts = vector(#Xi, i, vector(#V, j, tdlocal(Xi[i], A, B, Ls[i], V[j])));
  M = tdtheta(A, B, V, Xi, Ls, Pts);
  if(type(M) != "t_MAT" && #Xi, error("tdpairing: a local point is not one"));
  if(#Xi == 0, return([C, Pts, M, 0]));
  if(!tdalt(M), error("tdpairing: Theta_1 is not alternating: ", M));
  k = tdkernel(B, Xi, S, M);
  if(!k, error("tdpairing: the 2-torsion image ", B, " is not in the kernel of Theta_1"));
  [C, Pts, M, matrank(Mod(M, 2))];
}

\\ 1 if the class of d (in the span of the basis Xi) is in the kernel of the pairing matrix M
tdkernel(d, Xi, S, M) =
{
  my(G = Mod(Mat(apply(v -> tdclass(v, S)~, Xi)), 2), c = matinverseimage(G, Mod(tdclass(d, S)~, 2)));
  #c && c~ * Mod(M, 2) == 0;
}

\\ the verifier's check of a pairing claimed for Sel(A, B) with basis Xi: conic points C, local points Pts, rank rk.
\\ Returns the rank recomputed, or -1 if a check fails: the conic points lie on the conics, the local points are
\\ points with L != 0 (tdlclass), the matrix is alternating, B and A^2 - 4B are supported on S
tdvpairing(A, B, S, Xi, C, Pts, rk) =
{
  my(V = tdplaces(S), Ls, M);
  if(#C != #Xi || #Pts != #Xi || !tdsunit(B, S) || !tdsunit(A^2 - 4 * B, S), return(-1));
  for(i = 1, #Xi, my(P = C[i]);
    if(#P != 3 || P == [0, 0, 0] || Xi[i] * P[1]^2 + A * P[1] * P[2] + B / Xi[i] * P[2]^2 != P[3]^2, return(-1));
    if(#Pts[i] != #V, return(-1)));
  if(#Xi == 0, return(0));
  Ls = vector(#Xi, i, tdtangent(Xi[i], A, B, C[i]));
  M = tdtheta(A, B, V, Xi, Ls, Pts);
  if(type(M) != "t_MAT" || !tdalt(M), return(-1));
  matrank(Mod(M, 2));
}

\\ ---------- the Cassels-Tate pairing on Sel_2 (the third step)
\\ T. Fisher, On binary quartics and the Cassels-Tate pairing, Res. Number Theory 8 (2022) [F22], Theorem 3.1.
\\ E_{I,J}: y^2 = x^3 - 27 I x - 27 J; for a minimal model, I = c4 and J = 2 c6.  Sel_2(E) is the set of everywhere
\\ locally soluble binary quartics g = [a, b, c, d, e] (a x^4 + b x^3 z + ... + e z^4) with invariants I(g) = I, J(g) = J,
\\ up to proper equivalence [F22, Lemma 2.1].  L = Q[ph]/(ph^3 - 3 I ph + J), z(g) = (4 a ph + 3 b^2 - 8 a c)/3 in L,
\\ and [g1] + [g2] + [g3] = 0 iff z(g1) z(g2) z(g3) is a square in L.  With m^2 = z(g1) z(g2) z(g3), m in L*, and
\\ H1 the quadratic form (4) of g1, write (z(g2) z(g3)/m) H1 = alpha1 + beta1 ph + gamma1 ph^2; then, for points
\\ (x_v, z_v) with g1(x_v, z_v) a nonzero square in Q_v and gamma1(x_v, z_v) != 0, and g2(1, 0) != 0,
\\   <[g1], [g2]>_CT = prod over all places v of (g2(1, 0), gamma1(x_v, z_v))_v.
\\ Only finitely many v contribute [F22, Remark 3.3]: not those p >= 11 with g1, gamma1 p-integral, p not dividing
\\ disc(g1) content(gamma1) and g2(1, 0) a p-unit.  Here disc(g) = 16 (4 I^3 - J^2)/27 = 2^12 disc E is supported on
\\ S, so the places are V (tdplaces) and the primes of g2(1, 0) and of content(gamma1) outside it, which the
\\ certificate lists and the verifier proves prime.  The pairing is alternating, its kernel is the image of
\\ Sel_4(E) [F22; Cassels], and so rank E(Q) <= dim Sel_2(E) - t - rank CT, which bounds r_inf as well.  By
\\ [F17, Lemma 2.2] dim Sel_2(E) - t = dim S^(phi)(E) + dim S_2' - 2 = s' + s - rank Theta_1 - 2 (Theta_1 on Sel(a, b),
\\ the group of x-coordinates of E's points), so
\\   rank E(Q) <= s + s' - 2 - rank Theta_1 - rank CT_E,  and on E': rank E(Q) <= s + s' - 2 - rank Theta_1' - rank CT_E',
\\ and a family W in Sel_2 in place of all of it can only make rank CT smaller.

\\ the invariants of the quartic g
tdI(g) = 12 * g[1] * g[5] - 3 * g[2] * g[4] + g[3]^2;
tdJ(g) = 72 * g[1] * g[3] * g[5] - 27 * g[1] * g[4]^2 - 27 * g[2]^2 * g[5] + 9 * g[2] * g[3] * g[4] - 2 * g[3]^3;

\\ the variable of L, below every other
tdph = varlower("tdph");

\\ the cubic ph^3 - 3 I ph + J of L
tdcubic(I, J) = tdph^3 - 3 * I * tdph + J;

\\ z(g) in L = Q[ph]/F
tdz(g, F) = Mod((4 * g[1] * tdph + 3 * g[2]^2 - 8 * g[1] * g[3]) / 3, F);

\\ H(x, z) = (1/12) d^2 G/dx^2 + (2/9)(I - ph^2) z^2, G = (4 ph g + h)/3 (h the Hessian), as its coefficients of
\\ x^2, x z, z^2 in L
tdH(g, I, F) =
{
  my(a = g[1], b = g[2], c = g[3], d = g[4], e = g[5], h, G);
  h = [3*b^2 - 8*a*c, 4*(b*c - 6*a*d), 2*(2*c^2 - 24*a*e - 3*b*d), 4*(c*d - 6*b*e), 3*d^2 - 8*c*e];
  G = vector(5, k, Mod((4 * tdph * g[k] + h[k]) / 3, F));
  [G[1], G[2] / 2, G[3] / 6 + Mod(2 / 9 * (I - tdph^2), F)];
}

\\ the element m0 + m1 ph + m2 ph^2 of L, m = [m0, m1, m2]
tdL(m, F) = Mod(m[1] + m[2] * tdph + m[3] * tdph^2, F);

\\ gamma1 of (g1, g2, g3, m): the ph^2-coefficients of (z(g2) z(g3)/m) H1, a binary quadratic form [x^2, xz, z^2]
tdgamma(g1, g2, g3, m, I, F) =
{
  my(c = tdz(g2, F) * tdz(g3, F) / tdL(m, F));
  apply(h -> polcoef(lift(c * h), 2, tdph), tdH(g1, I, F));
}

\\ g(s, t) for the quartic g; f(s, t) for the quadratic form f
tdq4(g, s, t) = sum(k = 1, 5, g[k] * s^(5 - k) * t^(k - 1));
tdq2(f, s, t) = f[1] * s^2 + f[2] * s * t + f[3] * t^2;

\\ 1 if the rational v is a nonzero square in Q_p (p = 0: R)
tdissq(v, p) =
{
  my(k);
  if(v == 0, return(0));
  if(p == 0, return(v > 0));
  k = valuation(v, p); v /= p^k;
  if(k % 2, 0, p == 2, lift(Mod(v, 8)) == 1, kronecker(lift(Mod(v, p)), p) == 1);
}

\\ 1 if g is a quartic with invariants I, J, integral if integral = 1 (the g1 of tdctpair: its places come from
\\ disc(g1), so it must be integral; a g3 need not be)
tdquartic_ok(g, I, J, integral) = #g == 5 && tdI(g) == I && tdJ(g) == J && (!integral || denominator(g) == 1);

\\ 1 if the points Pts[k] show g locally soluble at the places V[k]
tdels(g, V, Pts) = #Pts == #V && #select(k -> !tdissq(tdq4(g, Pts[k][1], Pts[k][2]), V[k]), [1..#V]) == 0;

\\ <[g1], [g2]>_CT as 0 or 1, or -1 if a check fails: m^2 = z(g1) z(g2) z(g3) in L with m invertible, g2(1, 0) != 0, the
\\ extra primes X are primes in strictly increasing order (a prime listed twice would count its local symbol twice and
\\ flip the pairing), g2(1, 0) and content(gamma1) are supported on S u {3, 5, 7} u X, and Pts[k] is a point
\\ of g1 at the k-th place of V u X with gamma1 != 0
tdctpair(g1, g2, g3, m, I, F, S, X, Pts) =
{
  my(z = tdz(g1, F) * tdz(g2, F) * tdz(g3, F), M = tdL(m, F), gam, W, Q, a2 = g2[1], cg, e = 0);
  if(M^2 != z || !polresultant(lift(M), F) || !a2, return(-1));
  foreach(X, p, if(!isprime(p) || setsearch(tdplaces(S), p), return(-1)));
  for(k = 2, #X, if(X[k] <= X[k - 1], return(-1)));
  W = concat(tdplaces(S), X); Q = setunion(Set(concat(S, [3, 5, 7])), Set(X));
  gam = tdgamma(g1, g2, g3, m, I, F); cg = content(gam);
  if(!cg || !tdsunit(a2, Q) || !tdsunit(numerator(cg), Q) || !tdsunit(denominator(cg), Q) || #Pts != #W, return(-1));
  for(k = 1, #W, my(s = Pts[k][1], t = Pts[k][2], gv = tdq2(gam, s, t));
    if(!tdissq(tdq4(g1, s, t), W[k]) || !gv, return(-1));
    e += hilbert(a2, gv, W[k]) == -1);
  e % 2;
}

\\ the verifier's check of the Cassels-Tate data of the curve with minimal model mm: quartics Q with their local
\\ points Loc at V, and Pairs [i, j, g_ij, m_ij, local points of g_ij at V, extra primes, points] for every i < j.
\\ Returns the rank of the pairing matrix, or -1 if a check fails.
tdvct(mm, S, Q, Loc, Pairs) =
{
  my(E = ellinit(mm), I = E.c4, J = 2 * E.c6, F, V = tdplaces(S), n = #Q, M = matrix(n, n), seen = matrix(n, n));
  if(#Loc != n || #Pairs != n * (n - 1) / 2, return(-1));
  F = tdcubic(I, J);
  for(k = 1, n, if(!tdquartic_ok(Q[k], I, J, 1) || !tdels(Q[k], V, Loc[k]), return(-1)));
  foreach(Pairs, pr, my(i = pr[1], j = pr[2], v);
    if(i < 1 || j <= i || j > n || seen[i, j], return(-1));
    seen[i, j] = 1;
    if(!tdquartic_ok(pr[3], I, J, 0) || !tdels(pr[3], V, pr[5]), return(-1));
    v = tdctpair(Q[i], Q[j], pr[3], pr[4], I, F, S, pr[6], pr[7]);
    if(v < 0, return(-1));
    M[i, j] = M[j, i] = v);
  matrank(Mod(M, 2));
}

\\ ---------- the prover's Cassels-Tate data

\\ g(alpha x + beta z, gamma x + delta z), U = [alpha, beta; gamma, delta]
tdqsub(g, U) =
{
  my(q = sum(k = 1, 5, g[k] * (U[1, 1] * x + U[1, 2])^(5 - k) * (U[2, 1] * x + U[2, 2])^(k - 1)));
  vector(5, k, polcoef(q, 5 - k));
}

\\ g(x, z + k x) for the first k in 0, 1, -1, 2, ... with z(g) invertible in L and g(1, 0) != 0, preferring (if S != 0)
\\ g(1, 0) a prime times primes of S
tdnice(g, F, S) =
{
  my(first = 0);
  for(n = 0, 400, my(k = if(n % 2, (n + 1) \ 2, -(n \ 2)), h = tdqsub(g, [1, 0; k, 1]), a = h[1], r);
    if(!a || !polresultant(lift(tdz(h, F)), F), next);
    if(!first, first = h; if(S == 0, return(h)));
    r = abs(numerator(a)); for(i = 1, #S, r /= S[i]^valuation(r, S[i]));
    if(r == 1 || ispseudoprime(r), return(h)));
  if(!first, error("tdnice: no coordinates with z(g) invertible"));
  first;
}

\\ a quartic Q-equivalent to g but smaller: y^2 = g minimised and reduced by PARI (hyperellminimalmodel, hyperellred:
\\ y^2 + Q y = P, i.e. the quartic 4P + Q^2), then scaled by a rational square to the invariants I, J; g if that fails.
\\ Q-equivalent quartics with the same invariants are properly equivalent [F22, proof of Lemma 2.1], so the class is
\\ that of g (and the caller checks it again through m).
tdreduce(g, I, J) =
{
  my(h = iferr(hyperellred(hyperellminimalmodel(Pol(g * denominator(content(g))^2))), e, 0));
  if(!h, return(g));
  h = Vec(if(type(h) == "t_VEC", 4 * h[1] + h[2]^2, h)); h = concat(vector(5 - #h), h);
  foreach(tdmu2(h, I, J), mu2, if(issquare(mu2) && tdI(mu2 * h) == I && tdJ(mu2 * h) == J, return(mu2 * h)));
  g;
}

\\ a square root of z in L, from the square roots in the fields of the factors of F; 0 if z is not a square
tdsqrtL(z, F) =
{
  my(fa = factor(F)[, 1], parts = vector(#fa));
  for(k = 1, #fa, my(r = nfroots(fa[k], x^2 - lift(Mod(lift(z), fa[k]))));
    if(#r == 0, return(0));
    parts[k] = Mod(r[1], fa[k]));
  Mod(lift(chinese(parts)), F);
}

\\ a quartic with invariants I, J whose class has cubic invariant xi mod squares: write xi u^2 = Q0(u) + Q1(u) ph +
\\ Q2(u) ph^2 for u in L; on the conic Q2 = 0 (parametrised from a rational point) the quartic is Q1, scaled to the
\\ invariants I, J (by mu^2 with mu^4 = I/I(Q1), mu^6 = J/J(Q1)): the candidates, of which the caller keeps one in
\\ the right class (m^2 = z z z(g))
tdclassq(xi, I, J, F) =
{
  my(G = vector(3, c, matrix(3, 3)), sol, U, q, g);
  for(k = 0, 2, for(l = 0, 2, my(t = lift(xi * Mod(tdph^(k + l), F)));
    for(c = 0, 2, G[c + 1][k + 1, l + 1] = polcoef(t, c, tdph))));
  sol = qfsolve(G[3]);
  if(type(sol) != "t_COL", error("tdclassq: the conic has no rational point"));
  U = qfparam(G[3], sol) * [x^2, x, 1]~;
  q = U~ * G[2] * U;
  g = vector(5, k, polcoef(q, 5 - k));
  apply(mu2 -> mu2 * g, tdmu2(g, I, J));
}

\\ a real point [s, t] of g (g(s, t) > 0) with gamma(s, t) != 0 (gamma = 0: any): small points first, then rational
\\ approximations of points inside the intervals between real roots of g(x, 1) where g > 0
tdqreal(g, gam) =
{
  my(ok = (z -> gcd(z[1], z[2]) == 1 && tdq4(g, z[1], z[2]) > 0 && (gam == 0 || tdq2(gam, z[1], z[2]))), R, c);
  for(n = 1, 30, for(i = -n, n, my(z = [i, n - abs(i)]); if(ok(z), return(z))));
  localprec(100 + 2 * #Str(g));
  R = concat(concat([-1e40], polrootsreal(Pol(g * 1.))~), [1e40]);
  for(k = 1, #R - 1, my(mid = (R[k] + R[k + 1]) / 2);
    for(b = 1, 300, c = bestappr(mid, 2^b); c = [numerator(c), denominator(c)]; if(ok(c), return(c))));
  error("tdqreal: no real point on ", g, " with ", gam);
}

\\ a point [s, t] of g at the place p with gamma(s, t) != 0 (gamma = 0: any point of g)
tdqpoint(g, p, gam) =
{
  my(w, s, t, K, gi = g * denominator(content(g))^2);
  if(p == 0, return(tdqreal(g, gam)));
  w = tdwit(Pol(gi), p);
  if(type(w) != "t_VEC", s = w; t = 1,
    w = tdwit(subst(Polrev(gi), x, p * x), p);
    if(type(w) == "t_VEC", error("tdqpoint: the quartic is not soluble at ", p));
    s = 1; t = p * w);
  K = valuation(tdq4(gi, s, t), p) + 4;
  for(j = 0, 200, my(s1 = s + if(t == 1, j * p^K, 0), t1 = t + if(t == 1, 0, j * p^K));
    if(tdissq(tdq4(g, s1, t1), p) && (gam == 0 || tdq2(gam, s1, t1)), return([s1, t1])));
  error("tdqpoint: no point with gamma1 != 0 at ", p);
}

\\ the rationals mu^2 with I(mu^2 g) = mu^4 I(g) = I and J(mu^2 g) = mu^6 J(g) = J (two when J = 0)
tdmu2(g, I, J) =
{
  my(r);
  if(I && J, return([(J / tdJ(g)) / (I / tdI(g))]));
  if(!I, return(if(ispower(J / tdJ(g), 3, &r), [r], [])));
  if(issquare(I / tdI(g), &r), [r, -r], []);
}

\\ g scaled by a rational square mu^2 to the invariants I, J, integral; an error if that is not possible (then the
\\ cover is not of this model)
tdscale(g, I, J) =
{
  foreach(tdmu2(g, I, J), mu2, if(issquare(mu2) && denominator(mu2 * g) == 1 && tdI(mu2 * g) == I
                                    && tdJ(mu2 * g) == J, return(mu2 * g)));
  error("tdscale: the quartic has other invariants");
}

\\ the Cassels-Tate data of the curve with minimal model mm (S: 2 and its bad primes): [quartics, their local points at
\\ V, the pairs [i, j, g_ij, m_ij, local points of g_ij, extra primes, points], the matrix, its rank].  The quartics
\\ are PARI's ell2cover (a basis of Sel_2) in coordinates from tdnice.  Checks (a failure is a bug): every pair also in
\\ the order (j, i), the same value; tdvct accepts the data.
tdct(mm, S) =
{
  my(E = ellinit(mm), I = E.c4, J = 2 * E.c6, F = tdcubic(I, J), V = tdplaces(S), Q, Loc, Pairs = List(), n, M, R);
  Q = apply(c -> tdscale(concat(vector(5 - #Vec(c[1])), Vec(c[1])), I, J), ell2cover(E));
  Q = apply(g -> tdnice(g, F, S), Q); n = #Q;
  Loc = apply(g -> apply(p -> tdqpoint(g, p, 0), V), Q);
  M = matrix(n, n);
  for(i = 1, n, for(j = i + 1, n,
    my(zij = tdz(Q[i], F) * tdz(Q[j], F), g3, m = 0, mv, v = vector(2));
    foreach(tdclassq(zij, I, J, F), h, if(!m, g3 = tdnice(tdreduce(h, I, J), F, 0); m = tdsqrtL(zij * tdz(g3, F), F)));
    if(!m, error("tdct: no quartic for the sum of ", [i, j]));
    mv = vector(3, k, polcoef(lift(m), k - 1, tdph));
    for(o = 1, 2, my(g1 = if(o == 1, Q[i], Q[j]), g2 = if(o == 1, Q[j], Q[i]), gam = tdgamma(g1, g2, g3, mv, I, F), X, Pts);
      X = setminus(Set(concat([factor(abs(g2[1]))[, 1]~, factor(abs(numerator(content(gam))))[, 1]~,
                                factor(denominator(content(gam)))[, 1]~])), Set(V));
      Pts = apply(p -> tdqpoint(g1, p, gam), concat(V, X));
      v[o] = tdctpair(g1, g2, g3, mv, I, F, S, X, Pts);
      if(o == 1, listput(Pairs, [i, j, g3, mv, apply(p -> tdqpoint(g3, p, 0), V), X, Pts])));
    if(v[1] < 0 || v[1] != v[2], error("tdct: the pairing of ", [i, j], " is ", v));
    M[i, j] = M[j, i] = v[1]));
  Pairs = Vec(Pairs);
  R = matrank(Mod(M, 2));
  if(tdvct(mm, S, Q, Loc, Pairs) != R, error("tdct: the data do not check"));
  [Q, Loc, Pairs, M, R];
}

\\ ---------- the bound

\\ [B, S] with B = the least of the bounds Bs of the isogenies, S the largest integer <= B with (-1)^S = w
tdbound(Bs, w) =
{
  my(B = vecmin(Bs));
  [B, if((B - (w < 0)) % 2, B - 1, B)];
}

\\ ---------- the lower bound: characters of points
\\ For an odd prime p not dividing disc E and a root eps of T mod p, chi(P) = ((X(P) - eps) / p), with chi(P) = 1 if
\\ X(P) is not p-integral (P reduces to O) and chi(P) = (T'(eps) / p) if X(P) = eps mod p (then P reduces to (eps, 0),
\\ and T'(eps) = Q(eps), Q = T / (X - eps)).  This is the Kummer map of the 2-isogeny of E mod p with kernel
\\ (eps, 0), a homomorphism E(Q) -> {+-1} that kills 2E(Q).  As E(Q)/2E(Q) has dimension rank + t, and the torsion
\\ generators span E(Q)_tors / 2, t + r rows (the t torsion generators and r points) of rank t + r over F_2 prove
\\ that the r points are independent: rank E(Q) >= r.

\\ chi_{p, eps}(P) as 0 (for +1) or 1 (for -1); E any integral model, T its cubic, P a point of E or [0]
tdchi(E, T, P, p, eps) =
{
  my(X, k);
  if(P == [0], return(0));
  X = 4 * P[1];
  if(denominator(X) % p == 0, return(0));
  k = kronecker(lift(Mod(X - eps, p)), p);
  if(k == 0, k = kronecker(lift(Mod(subst(deriv(T), x, eps), p)), p));
  k == -1;
}

\\ the rows of the characters at the primes PR (every root of T mod each p): one row per point of PTS
tdchars(E, T, PTS, PR) =
{
  my(cols = List());
  foreach(PR, p, foreach(tdroots(T, p), eps, listput(cols, vector(#PTS, i, tdchi(E, T, PTS[i], p, eps))~)));
  if(#cols == 0, matrix(#PTS, 0), Mat(Vec(cols)));
}

\\ [the torsion generators of E (elltors) followed by PTS, the F_2-rank of their characters at PR]
tdrank(E, T, PTS, PR) =
{
  my(R = concat(elltors(E)[3], PTS), M = tdchars(E, T, R, PR));
  [R, if(#M, matrank(Mod(M, 2)), 0)];
}

\\ ---------- the prover

\\ the first primes, from PR and then up to pmax, at which the characters of the torsion generators and the points
\\ PTS (on the model ai) have rank t + #PTS; [] if PTS = [], and an error if pmax is not enough
tdprimes(ai, PTS, PR, pmax) =
{
  my(E = ellinit(ai), T = x^3 + E.b2*x^2 + 8*E.b4*x + 16*E.b6, R = concat(elltors(E)[3], PTS), M = matrix(#R, 0),
     keep = List(), rk = 0);
  if(#PTS == 0, return([]));
  PR = concat(PR, setminus(Set(primes([3, pmax])), Set(PR)));
  foreach(PR, p, my(N, k);
    if(!isprime(p) || p < 3 || E.disc % p == 0, next);
    N = matconcat([M, tdchars(E, T, R, [p])]); k = matrank(Mod(N, 2));
    if(k > rk, M = N; rk = k; listput(keep, p));
    if(rk == #R, return(Vec(keep))));
  error("The points do not prove rank >= ", #PTS, ": with the torsion, their characters have rank ", rk, " < ", #R,
        " at the primes up to ", pmax, ".");
}

\\ [points, primes] for the lower bound: the points PTS if their characters reach rank t + r at primes up to 10^4,
\\ else a 2-saturation of them (ellsaturation; independent points whose images in E(Q)/2E(Q) can be dependent
\\ until saturated), with primes up to pmax.  Only the characters are proof: the verifier checks them.
tdlower(ai, PTS, PR, pmax) =
{
  my(E = ellinit(ai), Q, pr = iferr(tdprimes(ai, PTS, PR, min(pmax, 10^4)), e, 0));
  if(pr != 0, return([PTS, pr]));
  Q = iferr(ellsaturation(E, PTS, 2), e, error("The points do not prove rank >= ", #PTS, ": ellsaturation: ",
                                                    errname(e), "."));
  [Q, tdprimes(ai, Q, PR, pmax)];
}

\\ everything the certificate states, as lines for rankbound/two_descent.py: ai the model, P its bad primes, PTS
\\ points on it (or []), PR primes to try first for them, pmax the search limit, r0 the known lower bound on the rank:
\\ the Cassels-Tate pairings are computed (on E, then on each E') only while S > r0
tdprove(ai, P, PTS, PR, pmax, r0) =
{
  my(M, E, S = Set(concat([2], P)), w, D, Th, Bs, BS, CE = 0, CI);
  foreach(P, p, if(p < 2 || !ispseudoprime(p), error("The bad prime ", p, " is not prime.")));
  if(#Set(P) != #P, error("The bad prime ", select(p -> #select(q -> q == p, P) > 1, P)[1], " is listed twice."));
  M = tdmodel(ai, P); E = M[1];
  foreach(P, p, if(E.disc % p, error("The bad prime ", p, " does not divide the discriminant.")));
  if(abs(E.disc) != prod(i = 1, #P, P[i]^valuation(E.disc, P[i])),
    error("The bad primes given do not account for the whole discriminant."));
  if(#M[3] == 0, error("tdprove: E(Q)[2] = 0"));
  w = ellrootno(E);
  D = vector(#M[3], i, tdisogeny(M[2], M[3][i], S));
  Th = vector(#D, i, my(d = D[i], ab2 = tdisog(d[2..3])); [tdpairing(d[2], d[3], S, d[4]), tdpairing(ab2[1], ab2[2], S, d[5])]);
  Bs = vector(#D, i, D[i][8] - Th[i][1][4] - Th[i][2][4]);
  CI = vector(#D);
  if(tdbound(Bs, w)[2] > r0,
    CE = iferr(tdct(E[1..5], S), e, print("CTSKIP\tE\t", errname(e), ": ", Str(e)); 0);
    if(CE, Bs = vector(#D, i, min(Bs[i], D[i][8] - Th[i][1][4] - CE[5]))));
  for(i = 1, #D, if(tdbound(Bs, w)[2] <= r0, break);
    my(ab2 = tdisog(D[i][2..3]), E2 = ellminimalmodel(ellinit([0, ab2[1], 0, ab2[2], 0])));
    CI[i] = iferr(tdct(E2[1..5], S), e, print("CTSKIP\t", i, "\t", errname(e), ": ", Str(e)); 0);
    if(CI[i], Bs[i] = min(Bs[i], D[i][8] - Th[i][2][4] - CI[i][5])));
  BS = tdbound(Bs, w);
  print("MM\t", strjoin(apply(z -> Str(z), Vec(E[1..5])), ","));
  print("TOR\t", strjoin(apply(z -> Str(z), elltors(E)[2]), ","));
  print("W\t", w);
  print("CN\t", ellglobalred(E)[1]);
  foreach(D, d, print("ISOG\t", d[1], "\t", d[2], "\t", d[3], "\t", strjoin(apply(z -> Str(z), d[4]), ","), "\t",
                      strjoin(apply(z -> Str(z), d[5]), ","), "\t", d[6], "\t", d[7], "\t", d[8], "\t", d[9], "\t", d[10]));
  for(i = 1, #D, for(g = 1, 2, my(T = Th[i][g]);
    for(k = 1, #T[1], print("PC\t", i, "\t", g, "\t", strjoin(apply(z -> Str(z), T[1][k]), ","), "\t",
                            strjoin(apply(z -> Str(z[1], ":", z[2]), T[2][k]), ","))));
    print("TH\t", i, "\t", Th[i][1][4], "\t", Th[i][2][4], "\t", Bs[i]);
    if(CI[i], print("CTI\t", i, "\t", CI[i][5], "\t", CI[i][1..3])));
  if(CE, print("CTE\t", CE[5], "\t", CE[1..3]));
  print("BS\t", BS[1], "\t", BS[2]);
  if(#PTS, my(L = tdlower(ai, PTS, PR, pmax));
    foreach(L[1], z, print("PT\t", z[1], "\t", z[2]));
    print("PR\t", strjoin(apply(z -> Str(z), L[2]), ",")));
}

\\ ---------- the verifier
\\ Prints "CHK name 0|1 info" per check (vchk, verify.gp) and "TD B S w": the recomputed bound.  ai: the
\\ certificate's model; P its bad primes; mm the claimed minimal model; tor the claimed torsion structure; D the
\\ claimed descents [e, a, b, Sel(a, b), Sel(a', b'), s, s', B_e, conic points, local points and rank of Theta_1 on
\\ Sel(a, b), the same for Theta_1' on Sel(a', b'), the Cassels-Tate data of E' (or 0)]; CE the Cassels-Tate data of
\\ E (or 0), each [quartics, their local points, pairs, rank] (tdvct)

tdverify(ai, P, mm, tor, D, CE) =
{
  my(M, E, Dis, S, R, w, ok, BS, got = List(), rkE = -1);
  foreach(P, p, if(!isprime(p), error("verify: ", p, " is not prime")));
  vchk("bad_primes_prime", 1, Str(#P, " primes, isprime (APRCL)"));
  M = tdmodel(ai, P); E = M[1]; Dis = E.disc;
  vchk("minimal_model", E[1..5] == mm);
  vchk("disc_factored", abs(Dis) == prod(i = 1, #P, P[i]^valuation(Dis, P[i])) && #Set(P) == #P
       && #select(p -> Dis % p, P) == 0, "the bad primes are exactly the primes of the minimal discriminant");
  vchk("torsion", elltors(E)[2] == tor, Str("E(Q)_tors = ", elltors(E)[2]));
  R = M[3];
  vchk("two_torsion", #R && apply(d -> d[1], D) == R, Str("the roots of T: ", R));
  S = Set(concat([2], P));
  ok = #R && #D == #R;
  if(CE, rkE = tdvct(E[1..5], S, CE[1], CE[2], CE[3]);
    vchk("cassels_tate", rkE >= 0 && rkE == CE[4], Str("rank of the Cassels-Tate pairing on Sel_2(E): ", rkE)));
  for(i = 1, #D, if(!ok, break);
    my(d = D[i], c = tdisogeny(M[2], d[1], S), ab2 = tdisog(c[2..3]), r1, r2, sp, Be);
    sp = c[2..3] == d[2..3] && tdsamespan(d[4], c[4], S) && tdsamespan(d[5], c[5], S) && c[6..7] == d[6..7];
    vchk(Str("descent_", i), sp, Str("e = ", d[1], ": s = ", c[6], ", s' = ", c[7]));
    vchk(Str("local_duality_", i), c[9] && c[10], "W_v and W'_v are orthogonal complements; b, b' in the Selmer groups");
    \\ the pairings on the certificate's bases (checked above to span the Selmer groups)
    [r1, r2] = if(sp, [tdvpairing(c[2], c[3], S, d[4], d[9], d[10], d[11]),
                       tdvpairing(ab2[1], ab2[2], S, d[5], d[12], d[13], d[14])], [-1, -1]);
    vchk(Str("pairing_", i), r1 == d[11] && r2 == d[14], Str("rank Theta_1 = ", r1, ", rank Theta_1' = ", r2));
    Be = c[8] - r1 - r2;
    if(rkE >= 0, Be = min(Be, c[8] - r1 - rkE));
    if(d[15], my(E2 = ellminimalmodel(ellinit([0, ab2[1], 0, ab2[2], 0])), rk = tdvct(E2[1..5], S, d[15][1], d[15][2], d[15][3]));
      vchk(Str("cassels_tate_", i), rk >= 0 && rk == d[15][4], Str("rank of the Cassels-Tate pairing on Sel_2(E'): ", rk));
      Be = min(Be, c[8] - r2 - rk));
    vchk(Str("bound_", i), Be == d[8], Str("B_e = ", Be));
    listput(got, Be));
  w = ellrootno(E);
  if(ok, BS = tdbound(Vec(got), w); print("TD ", BS[1], " ", BS[2], " ", w));
}

\\ the lower bound check: the points PTS on the model ai, with the characters at the primes PR
tdvrank(ai, r, PTS, PR) =
{
  my(E = ellinit(ai), T = x^3 + E.b2*x^2 + 8*E.b4*x + 16*E.b6, k = #elltors(E)[3], ok);
  ok = #PTS == r && #Set(PR) == #PR;
  for(i = 1, #PTS, ok = ok && ellisoncurve(E, PTS[i]));
  foreach(PR, p, ok = ok && isprime(p) && p >= 3 && E.disc % p);
  vchk("rank_lower_bound", ok && (r == 0 || tdrank(E, T, PTS, PR)[2] == k + r),
       Str(r, " points and ", k, " torsion generators, characters at ", #PR, " primes"));
}
