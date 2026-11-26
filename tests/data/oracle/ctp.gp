\\ The prover's half of the Cassels-Tate pairing on Sel_2(E) (verify/ctp.gp has the mathematics and the evaluation,
\\ tdctpair, and the check, tdvct): binary quartics in given classes, their local points, the quartics of the sums,
\\ the pairing matrix.  Used by the 2-isogeny route (tower.gp's tdprove: tdct on ell2cover's quartics) and by M2
\\ (ctprove below: curves with E(Q)[2] = 0, quartics from M1's Selmer elements).  Loaded after verify/core.gp,
\\ verify/ctp.gp and verify/isogeny.gp (tdsquare, for tdsqwit); M2 also after relations/split.gp (m1K, m1el).

\\ ---------- the prover's local points (also tower.gp's, for the isogeny pairings)

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

\\ ---------- the quartics and the pairing matrix

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
  my(h = iferr(hyperellred(hyperellminimalmodel(Pol(g * denominator(content(g))^2))), e, 0, errname(e) != "e_ALARM"));
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
\\ the right class (m^2 = z z z(g)).  fa = 0: qfsolve factors the conic's determinant; fa = [P0, cof]: tdqfsolve with
\\ the primes P0 and the cofactors cof (fake primes)
tdclassq(xi, I, J, F, fa = 0) =
{
  my(G = vector(3, c, matrix(3, 3)), sol, U, q, g);
  for(k = 0, 2, for(l = 0, 2, my(t = lift(xi * Mod(tdph^(k + l), F)));
    for(c = 0, 2, G[c + 1][k + 1, l + 1] = polcoef(t, c, tdph))));
  sol = if(fa, tdqfsolve(G[3], fa[1], fa[2]), qfsolve(G[3]));
  if(type(sol) != "t_COL", error("tdclassq: the conic has no rational point"));
  U = qfparam(G[3], sol) * [x^2, x, 1]~;
  q = U~ * G[2] * U;
  g = vector(5, k, polcoef(q, 5 - k));
  apply(mu2 -> mu2 * g, tdmu2(g, I, J));
}

\\ the quartic of the class of xi in L (z(g) = xi mod squares): of the candidates of tdclassq (fa as there), the first
\\ in the class, reduced (tdreduce) and in coordinates from tdnice(., F, S); [g, m] with m^2 = xi z(g), m = 0 if no
\\ candidate is in the class.  The one routine for every quartic: M2's quartics of Selmer elements and both methods'
\\ quartics of the sums of two classes.
tdclassg(xi, I, J, F, S, fa = 0) =
{
  my(g, m = 0);
  foreach(tdclassq(xi, I, J, F, fa), h, if(!m, g = tdnice(tdreduce(h, I, J), F, S); m = tdsqrtL(xi * tdz(g, F), F)));
  [g, m];
}

\\ ---------- qfsolve with "fake primes" (E2 #1)
\\ The determinant of a class conic is the norm of its element up to primes of S and 3, and the part outside those is
\\ the square of a known number (N(D) for M2's elements, the n_i of tdcof for the sums), whose factorisation is not:
\\ qfsolve is given those numbers as if prime.  A composite one can make qfsolve fail; an impossible inverse shows a
\\ factor, which splits it; any other failure, or 8 of them, factors the pieces (each divides a known cofactor, so
\\ it is small), and qfsolve gets the true factorisation.  Every solution is checked by substitution.

\\ the coprime base of the integers > 1 in L (each element of L a product of powers of the base)
tdcoprime(L) =
{
  my(B = Set(select(z -> z > 1, L)), ch = 1);
  while(ch, ch = 0;
    for(i = 1, #B, for(j = i + 1, #B, my(g = gcd(B[i], B[j]));
      if(g > 1, B = Set(select(z -> z > 1, concat(vecextract(B, setminus([1..#B], [i, j])), [g, B[i] / g, B[j] / g]))); ch = 1; break(2)))));
  B;
}

\\ the factorisation matrix of the positive integer d over the base B (primes P0 first), with the rest of d (if not 1)
\\ as one more factor
tdfakefa(d, P0, B) =
{
  my(f = List(), r = d);
  foreach(concat(P0, B), q, my(v = valuation(r, q)); if(v, listput(f, [q, v]); r /= q^v));
  if(r > 1, listput(f, [r, 1]));
  matrix(#f, 2, i, k, f[i][k]);
}

\\ an isotropic vector of the nonsingular symmetric rational 3 x 3 matrix G whose determinant is, outside the primes
\\ P0, a product of powers of factors of the integers cof; or what qfsolve returns when there is none
tdqfsolve(G, P0, cof) =
{
  my(G0 = G * denominator(G), d = abs(matdet(G0)), B, s, r, ok);
  ok = (s -> type(s) == "t_COL" && s != 0 * s && s~ * G0 * s == 0);
  if(!d, return(qfsolve(G0)));
  P0 = Set(concat(P0, factor(denominator(G))[, 1]~));
  r = d; foreach(P0, p, r /= p^valuation(r, p));
  B = tdcoprime(concat(apply(c -> gcd(r, c), cof), [r]));
  for(t = 1, 8,
    s = iferr(qfsolve([G0, tdfakefa(d, P0, B)]), e, e, errname(e) != "e_ALARM");
    if(ok(s), return(s));
    my(z = if(type(s) == "t_ERROR" && errname(s) == "e_INV", component(s, 2), 0), B2 = B);
    if(type(z) == "t_INTMOD", B2 = tdcoprime(concat(B, concat(apply(b -> [gcd(lift(z), b), gcd(z.mod, b)], B)))));
    if(B2 == B, break);
    B = B2);
  \\ the true factorisation: each factor of the base factored (each divides one cofactor, so it is small)
  B = Set(concat(apply(b -> factor(b)[, 1]~, B)));
  qfsolve([G0, tdfakefa(d, P0, B)]);
}

\\ the number n with |N(z(g))| = n^2 times a product of primes of P0 (z(g) in L(S, 2): its non-S part is a square), or
\\ that part itself when it is not a square
tdcof(g, F, P0) =
{
  my(nz = norm(tdz(g, F)), r = abs(numerator(nz)) * denominator(nz), n);
  foreach(P0, p, r /= p^valuation(r, p));
  if(issquare(r, &n), n, r);
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
\\ are PARI's ell2cover (a basis of Sel_2) in coordinates from tdnice (2-isogeny route).
tdct(mm, S) =
{
  my(E = ellinit(mm), I = E.c4, J = 2 * E.c6, F = tdcubic(I, J), Q);
  Q = apply(c -> tdscale(concat(vector(5 - #Vec(c[1])), Vec(c[1])), I, J), ell2cover(E));
  tdctq(mm, S, apply(g -> tdnice(g, F, S), Q), 0);
}

\\ the Cassels-Tate data of the quartics Q (integral, ELS, invariants c4, 2 c6 of the minimal model mm), as tdct's: the
\\ local points, a quartic of each sum (tdclassg), every pairing.  fake: 1 to give qfsolve the primes of S and 3 and
\\ the cofactors of tdcof (M2), 0 to let it factor (the 2-isogeny route, as before).  Checks (a failure is a bug):
\\ every pair also in the order (j, i), the same value; tdvct accepts the data.
tdctq(mm, S, Q, fake) =
{
  my(E = ellinit(mm), I = E.c4, J = 2 * E.c6, F = tdcubic(I, J), V = tdplaces(S), Loc, Pairs = List(), n = #Q, M, R,
     P0 = setunion(S, [3]), cf);
  Loc = apply(g -> apply(p -> tdqpoint(g, p, 0), V), Q);
  if(fake, cf = apply(g -> tdcof(g, F, P0), Q));
  M = matrix(n, n);
  for(i = 1, n, for(j = i + 1, n,
    my(zij = tdz(Q[i], F) * tdz(Q[j], F), g3, m, mv, v = vector(2));
    [g3, m] = tdclassg(zij, I, J, F, 0, if(fake, [P0, [cf[i], cf[j]]], 0));
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

\\ ---------- M2: the pairing on the Selmer elements of M1 (curves with E(Q)[2] = 0; rankbound/ctp.py)
\\ K = Q(alpha) as in M1 (relations/split.gp's m1K: K[1] the nf, K[2] alpha, K[4] the primes of lc(F)); theta =
\\ q0 + q1 alpha + q2 alpha^2 with W(theta) = 0, W = x^3 + b2 x^2 + 8 b4 x + 16 b6.  Under x' = 36 X + 3 b2 = 3(3 x + b2)
\\ the model y^2 = x'^3 - 27 c4 x' - 54 c6 is E_{I,J} with I = c4, J = 2 c6, whose 2-division roots are x' = -3 ph,
\\ ph^3 - 3 I ph + J = 0; so K -> L, theta -> -(ph + b2)/3, carries x - theta to (x' + 3 ph)/9, the Kummer image of
\\ verify/ctp.gp's z (z(g) = xi mod squares for the quartic g of xi).  A Selmer element xi (a product of relations)
\\ is first replaced by a small element of its class mod squares: (xi) = A D^2 with A the primes of odd valuation
\\ (above S, as xi is in K(S, 2)) and D a fractional ideal; products of two such are reduced by b in D^-1 with
\\ sum_v |xi|_v |b|_v^2 least (LLL in the twisted T2 norm), xi -> xi b^2, D -> b D; leaves first, then pairs, in a
\\ balanced tree, so no product of many relations is ever formed.  (PARI's idealredmodpower takes no factored form
\\ and does not reduce the unit part; this does both, from the relations' known primes.)

CTDEC = Map();
\\ the prime ideals above p, cached
ctdec(nf, p) = my(d); if(mapisdefined(CTDEC, p, &d), d, d = idealprimedec(nf, p); mapput(CTDEC, p, d); d);

\\ [y, A, D] of the relation r (split.gp's form: [0, a, b], [1, c0, c1, c2, d], [2, p]) with norm primes H: the
\\ element y (on nf's integral basis), A the set of the primes P with v_P(y) odd, D = prod P^floor(v_P / 2); an
\\ error if the primes of H, lc(F) and d do not account for N(y)
ctleaf(K, r, H) =
{
  my(nf = K[1], y = nfalgtobasis(nf, m1el(K, r)), ps = Set(concat(H, K[4])), A = List(), D = 1, nn = 1);
  if(r[1] == 1, ps = setunion(ps, factor(r[5])[, 1]~));
  if(r[1] == 2, ps = setunion(ps, [r[2]]));
  foreach(ps, p, foreach(ctdec(nf, p), pr, my(v = nfeltval(nf, y, pr));
    if(v, nn *= p^(pr.f * v); if(v % 2, listput(A, pr));
      if(floor(v / 2), D = idealmul(nf, D, idealpow(nf, pr, floor(v / 2)))))));
  if(abs(nfeltnorm(nf, y)) != nn, error("ctleaf: the primes of relation ", r, " do not give its norm"));
  [y, Set(Vec(A)), D];
}

\\ the binary size of the rationals in the vector or matrix v
ctbits(v) = my(m = 1); foreach(concat(Vec(v), []), c, foreach(Vec(c), z, m = max(m, max(abs(numerator(z)), denominator(z))))); exponent(m);

\\ the embeddings of z (on nf's integral basis) at the roots R of nf.pol
ctemb(nf, z, R) = my(p = lift(nfbasistoalg(nf, z))); vector(#R, k, subst(p, variable(nf.pol), R[k]));

\\ [y b^2, A, b D] for b in D^-1 with sum_v c_v |y|_v |b|_v^2 least (c_v = 1, 2 at real, complex places): LLL in that
\\ twisted T2 norm (the plain T2 norm if it fails numerically), the embeddings at a precision from the sizes of y,
\\ D^-1 and the roots (the products' cancellation)
ctred(K, t) =
{
  my(nf = K[1], y = nfalgtobasis(nf, t[1]), D = t[3], n = poldegree(nf.pol), r1 = nf.sign[1], Bm, R, w, Em, Gm, U, b,
     gm, c);
  Bm = idealhnf(nf, idealinv(nf, D));
  localbitprec(128 + 2 * (ctbits(y) + ctbits(Bm)) + 4 * n * exponent(1 + vecmax(abs(polroots(nf.pol)))));
  R = polroots(nf.pol); R = concat(real(select(z -> imag(z) == 0, R)), select(z -> imag(z) > 0, R));
  if(#R != r1 + nf.sign[2], error("ctred: the roots of nf.pol"));
  c = vector(#R, k, if(k <= r1, 1, 2));
  w = abs(ctemb(nf, y, R));
  gm = exp(sum(k = 1, #w, c[k] * log(w[k])) / n);
  w = w / gm;
  Em = matrix(#R, n, k, j, ctemb(nf, Bm[, j], R)[k]);
  Gm = matrix(n, n, i, j, sum(k = 1, #w, c[k] * w[k] * real(Em[k, i] * conj(Em[k, j]))));
  U = iferr(qflllgram(Gm), e, 0, errname(e) != "e_ALARM");
  if(type(U) != "t_MAT" || #U < n, U = qflllgram(matrix(n, n, i, j, sum(k = 1, #w, c[k] * real(Em[k, i] * conj(Em[k, j]))))));
  b = Bm * U[, 1];
  [nfalgtobasis(nf, nfeltmul(nf, y, nfeltpow(nf, b, 2))), t[2], idealmul(nf, b, D)];
}

\\ the product of two reduced [y, A, D], reduced: A1 + A2 mod 2, the primes of both into D
ctmul(K, s, t) =
{
  my(nf = K[1], C = setintersect(s[2], t[2]), D = idealmul(nf, s[3], t[3]));
  foreach(C, pr, D = idealmul(nf, D, pr));
  ctred(K, [nfeltmul(nf, s[1], t[1]), setminus(setunion(s[2], t[2]), C), D]);
}

\\ the reduced product of the leaves L[a..b], a balanced tree
ctprod(K, L, a, b) = if(a == b, ctred(K, L[a]), my(c = (a + b) \ 2); ctmul(K, ctprod(K, L, a, c), ctprod(K, L, c + 1, b)));

\\ the image in L of the generator of nf (K[1].pol's root): theta = th(alpha) = h(gen) (modreverse: gen = h'(theta)),
\\ theta -> -(ph + b2)/3
ctgenL(K, th, b2, F) =
{
  my(t = th[1] + th[2] * K[2] + th[3] * K[2]^2, h = modreverse(t));
  Mod(subst(lift(h), variable(h.mod), (-tdph - b2) / 3), F);
}

\\ the element y of K (on nf's integral basis) in L, gL the image of nf's generator (ctgenL)
ctL(K, y, gL) = subst(lift(nfbasistoalg(K[1], y)), variable(K[1].pol), gL);

\\ the characters of M1 (format 4: F_2, 1 for -1) of x - theta for the rational x: the signs at the real roots RR of
\\ F(x, 1), then the Legendre symbols at the primes [q, c] (alpha -> c); -1 where undefined (q | x - theta(c))
ctchars(x0, th, RR, CQ) =
{
  my(s = vector(#RR, k, x0 - (th[1] + th[2] * RR[k] + th[3] * RR[k]^2) < 0), q);
  concat(s, vector(#CQ, k, my(q = CQ[k][1], c = CQ[k][2], d = denominator(x0) * denominator(th), v);
    if(d % q == 0, -1, v = lift(Mod(x0, q) - (Mod(th[1], q) + Mod(th[2], q) * c + Mod(th[3], q) * c^2));
      if(v == 0, -1, kronecker(v, q) < 0))));
}

\\ the indices of a complement in the Selmer basis (rows of Ech, their characters) of the span of the points' images:
\\ each point x (an x-coordinate of y^2 = W(x)) is located by its characters (where defined); a point not in the span
\\ is ignored.  [the complement, the number of points located, the rank of their span]
ctcomplement(f, th, X, Ech, nr, CQ) =
{
  my(d = #Ech[, 1], RR, rows = List(), Pm, J);
  if(#X == 0 || d == 0, return([[1..d], 0, 0]));
  localprec(200); RR = vecsort(real(select(z -> imag(z) == 0, polroots(f))));
  if(#RR != nr, error("ctcomplement: ", #RR, " real roots, not ", nr));
  foreach(X, x0, my(pc = ctchars(x0, th, RR, CQ), cols = select(k -> pc[k] >= 0, [1..#pc]), c);
    c = matinverseimage(Mod(vecextract(Ech, [1..d], cols), 2)~, Mod(vecextract(pc, cols)~, 2));
    if(#c, listput(rows, lift(c)~)));
  if(#rows == 0, return([[1..d], 0, 0]));
  Pm = Mod(Mat(Vec(rows)~), 2);
  if(matrank(Pm) == 0, return([[1..d], #rows, 0]));
  J = matindexrank(Pm)[2];
  [setminus([1..d], Set(J)), #rows, matrank(Pm)];
}

\\ M2's data, as lines for rankbound/ctp.py: the curve's minimal model mm, bad primes P; M1's index form f (a
\\ polynomial in x), theta th, relations R (split.gp's form) with their norm primes H, the basis of Sel_2 cap V as
\\ index sets SelJ of relations (1-based) with their characters Ech (rows; nr signs, then the primes CQ); the points'
\\ x-coordinates X on y^2 = W(x).  The quartics of a complement C of the points' span, every pairing on them, and a
\\ subset W of C with nonsingular Gram matrix of the same rank: the witness's section (CTE), checked by tdvct.
\\ Lines: CTC (#basis, points located, their rank, C), CTQ (ms for the quartics), CTG (ms, the Gram rank), CTE.
ctprove(mm, P, f, th, R, H, SelJ, Ech, nr, CQ, X) =
{
  my(E = ellinit(mm), I = E.c4, J = 2 * E.c6, F = tdcubic(I, J), S = Set(concat([2], P)), P0 = setunion(S, [3]), K, gL,
     C, Q = List(), leaves = Map(), t0 = getabstime(), res, M, R2, Wi, pos, QW, LW, PW);
  addprimes(P);
  K = m1K(f, P);
  gL = ctgenL(K, th, E.b2, F);
  C = ctcomplement(f, th, X, Ech, nr, CQ);
  print("CTC\t", #SelJ, "\t", C[2], "\t", C[3], "\t", strjoin(apply(z -> Str(z - 1), C[1]), ","));
  foreach(C[1], t, my(L, y, xi, g, m);
    L = vector(#SelJ[t], k, my(j = SelJ[t][k], lf);
      if(!mapisdefined(leaves, j, &lf), lf = ctleaf(K, R[j], H[j]); mapput(leaves, j, lf)); lf);
    y = ctprod(K, L, 1, #L);
    if(#select(pr -> !setsearch(S, pr.p), y[2]), error("ctprove: a Selmer element has an odd valuation outside S"));
    xi = ctL(K, y[1], gL);
    [g, m] = tdclassg(xi, I, J, F, S, [P0, [idealnorm(K[1], y[3])]]);
    if(!m, error("ctprove: no quartic in the class of a Selmer element"));
    if(denominator(g) != 1, error("ctprove: the reduced quartic of a Selmer element is not integral: ", g));
    listput(Q, g));
  Q = Vec(Q);
  print("CTQ\t", getabstime() - t0); t0 = getabstime();
  if(#Q < 2, print("CTG\t0\t0"); return);
  res = tdctq(mm, S, Q, 1);
  M = res[4]; R2 = res[5];
  print("CTG\t", getabstime() - t0, "\t", R2);
  if(!R2, return);
  Wi = vecsort(Vec(matindexrank(Mod(M, 2))[1]));
  if(#Wi != R2 || matrank(Mod(vecextract(M, Wi, Wi), 2)) != R2, error("ctprove: the Gram matrix on W is singular"));
  pos = vector(#Q); for(k = 1, #Wi, pos[Wi[k]] = k);
  QW = vecextract(Q, Wi); LW = vecextract(res[2], Wi);
  PW = [concat([pos[pr[1]], pos[pr[2]]], pr[3..7]) | pr <- res[3], pos[pr[1]] && pos[pr[2]]];
  if(tdvct(mm, S, QW, LW, PW) != R2, error("ctprove: the data on W do not check"));
  print("CTE\t", R2, "\t", [QW, LW, PW]);
}
