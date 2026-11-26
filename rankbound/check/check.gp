\\ The checker's checks in PARI/GP.  The Python passes every value as a JSON integer or a canonical decimal (so a
\\ witness cannot run gp code) and reads back one line "CHK name 0|1 info" per check, plus the data lines named below.
\\   1. shared: the output line, the curve, the rank lower bound from points (vchk, vcurve, vtors, vrootno, vrank)
\\   2. the Cassels-Tate pairing on Sel_2(E) by binary quartics, for both families of witnesses (tdvct)
\\   3. curves with a rational point of order 2: the 2-isogeny descent (tdverify)
\\   4. curves with E(Q)[2] = 0: the field and its bound (vfield, v4field), the relations (ckrels), the local data and
\\      the classes (v4main), the Cassels-Tate section (v4ct)

\\ ==================== 1. shared

vchk(name, ok, info = "") = print("CHK ", name, " ", if(ok, 1, 0), " ", info);

\\ Stop after a failed check when what follows would work on unconfirmed data (such as a B capped by a claimed d_K);
\\ the checks not printed then fail (need, in the Python).
vhalt() = print("HALT"); quit(0);

\\ Check the curve: P are the primes of the minimal discriminant, proved prime, and mm is the minimal model of ai.
\\ Return the minimal model, with P in PARI's prime table.  That each p divides disc(ai) is tested before APRCL.
vcurve(ai, P, mm) =
{
  my(E, D, D0 = ellinit(ai).disc);
  foreach(P, p, if(D0 % p, vchk("disc_factored", 0, "a bad prime does not divide the discriminant"); vhalt()));
  foreach(P, p, if(!isprime(p), error("the bad prime ", p, " is not prime")));
  vchk("bad_primes_prime", 1, Str(#P, " primes, isprime (APRCL)"));
  addprimes(P); E = ellminimalmodel(ellinit(ai)); D = E.disc;
  vchk("minimal_model", E[1..5] == mm);
  vchk("disc_factored", abs(D) == prod(i = 1, #P, P[i]^valuation(D, P[i])) && #Set(P) == #P
       && #select(p -> D % p, P) == 0, "the bad primes are exactly the primes of the minimal discriminant");
  E;
}

\\ ai any model
vtors(ai, tor) = my(t = elltors(ellinit(ai))[2]); vchk("torsion", t == tor, Str("E(Q)_tors = ", t));

vrootno(E, wc) = my(w = ellrootno(E)); vchk("root_number", w == wc, Str("w = ", w)); w;

\\ ---------- the rank lower bound: r points independent in E(Q)/2E(Q)
\\ Characters chi: E(Q) -> F_2 that kill 2E(Q), at odd good primes: 2-signatures if E(Q)[2] = 0, else the Kummer maps
\\ of the 2-isogenies mod p.  0 stands for +1, and X = 4x.  A point reducing to O has chi = 0; one with X(P) = eps mod p
\\ has chi(P) = (T'(eps) / p).  The t torsion generators' rows come first: rank t + r proves the r points independent.

\\ the roots in [0, p) of H mod p (H mod p != 0)
tdroots(H, p) = apply(lift, Vec(polrootsmod(H, p)));

\\ chi_{p, eps}(P); E any integral model, T its cubic, P a point of E or [0]
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

\\ one row per point of PTS, one column per p in PR and root of T mod p
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

\\ Check r points PTS on the model ai, with the characters at the distinct odd good primes PR.
vrank(ai, r, PTS, PR) =
{
  my(E = ellinit(ai), T = x^3 + E.b2*x^2 + 8*E.b4*x + 16*E.b6, ok);
  ok = #PTS == r && #Set(PR) == #PR;
  for(i = 1, #PTS, ok = ok && ellisoncurve(E, PTS[i]));
  \\ r = 1 without primes (a cyclic cubic field): nP != O for n <= 12, so P has infinite order by Mazur's bound
  if(r == 1 && #PR == 0,
    return(vchk("rank_lower_bound", ok && prod(n = 1, 12, ellmul(E, PTS[1], n) != [0]), "1 point of infinite order")));
  if(polisirreducible(T), vrank2(E, r, PTS, PR, ok), vrankiso(E, T, r, PTS, PR, ok));
}

vrank2(E, r, PTS, PR, ok) =
{
  my(rows = []);
  for(k = 1, #PR, my(p = PR[k], Ep, n, h, row); ok = ok && isprime(p) && p >= 3 && E.disc % p; if(!ok, break);
    Ep = ellinit(E[1..5], p); n = ellcard(Ep); ok = ok && n % 4 == 2; if(!ok, break); h = n / 2;
    row = vector(#PTS, i, if(denominator(PTS[i][1]) % p == 0 || denominator(PTS[i][2]) % p == 0, 0,
      my(Q = [Mod(PTS[i][1], p), Mod(PTS[i][2], p)]); if(ellmul(Ep, Q, h) == [0], 0, 1)));
    rows = concat(rows, [row]));
  vchk("rank_lower_bound", ok && (r == 0 || (#rows && matrank(Mod(matconcat(rows~), 2)) == r)), Str(r, " points, 2-signatures at ", #PR, " primes"));
}

vrankiso(E, T, r, PTS, PR, ok) =
{
  my(k = #elltors(E)[3]);
  foreach(PR, p, ok = ok && isprime(p) && p >= 3 && E.disc % p);
  vchk("rank_lower_bound", ok && (r == 0 || tdrank(E, T, PTS, PR)[2] == k + r),
       Str(r, " points and ", k, " torsion generators, characters at ", #PR, " primes"));
}

\\ ==================== 2. the Cassels-Tate pairing on Sel_2(E) by binary quartics, for every curve E/Q
\\ Fisher's formula for the pairing [Fis22, Thm. 3.1, Remark 3.3].  g = [a, b, c, d, e] is
\\ a x^4 + b x^3 z + ... + e z^4, with I = c4 and J = 2 c6 of the minimal model; ph is the variable of L.  As
\\ disc(g) = 2^12 disc E, the places are tdplaces(S) and the witness's extra primes of g2(1, 0) and content(gamma1).

\\ the places that can contribute: 0 (R), the primes of S (2 among them), and 3, 5, 7
tdplaces(S) = concat([0], setunion(S, [3, 5, 7]));

\\ 1 if the integer n != 0 has all its prime factors in S
tdsunit(n, S) = my(m = abs(n)); if(!m, return(0)); foreach(S, p, m /= p^valuation(m, p)); m == 1;

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

\\ 1 if g is a quartic with invariants I, J, integral if integral = 1 (a g1 must be, as its places come from disc(g1))
tdquartic_ok(g, I, J, integral) = #g == 5 && tdI(g) == I && tdJ(g) == J && (!integral || denominator(g) == 1);

\\ 1 if the points Pts[k] show g locally soluble at the places V[k]
tdels(g, V, Pts) = #Pts == #V && #select(k -> !tdissq(tdq4(g, Pts[k][1], Pts[k][2]), V[k]), [1..#V]) == 0;

\\ <[g1], [g2]>_CT as 0 or 1, or -1 if a check fails.  The extra primes X must be strictly increasing, as a repeated
\\ prime would count its symbol twice; Pts[k] is a point of g1 at the k-th place of V u X.
tdctpair(g1, g2, g3, m, I, F, S, X, Pts) =
{
  my(z = tdz(g1, F) * tdz(g2, F) * tdz(g3, F), M = tdL(m, F), gam, W, Q, a2 = g2[1], cg, e = 0);
  if(M^2 != z || !polresultant(lift(M), F) || !a2, return(-1));
  foreach(X, p, if(p < 2 || setsearch(tdplaces(S), p), return(-1)));
  for(k = 2, #X, if(X[k] <= X[k - 1], return(-1)));
  W = concat(tdplaces(S), X); Q = setunion(Set(concat(S, [3, 5, 7])), Set(X));
  gam = tdgamma(g1, g2, g3, m, I, F); cg = content(gam);
  if(!cg || !tdsunit(a2, Q) || !tdsunit(numerator(cg), Q) || !tdsunit(denominator(cg), Q) || #Pts != #W, return(-1));
  \\ BPSW first (cheap, and true for every prime), so that the points are tested before APRCL
  foreach(X, p, if(!ispseudoprime(p), return(-1)));
  for(k = 1, #W, my(s = Pts[k][1], t = Pts[k][2], gv = tdq2(gam, s, t));
    if(!tdissq(tdq4(g1, s, t), W[k]) || !gv, return(-1));
    e += hilbert(a2, gv, W[k]) == -1);
  foreach(X, p, if(!isprime(p), return(-1)));
  e % 2;
}

\\ Return the rank of the pairing matrix of the quartics Q (points Loc), or -1 if a check fails.  Pairs: [i, j, g_ij,
\\ m_ij, points of g_ij, extra primes, points] for every i < j.
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

\\ ==================== 3. the 2-isogeny descent, for curves E/Q with a rational point of order 2
\\ T is monic and integral, so its rational roots are integers.  Outside
\\ S = {2} u the bad primes, p is odd and divides neither B nor A^2 - 4B, so the local image is the unit classes, which
\\ Q(S, 2) builds in; Sel is the kernel of the map Q(S, 2) -> prod_v (Q_v*/Q_v*^2) / W_v.

\\ ---------- the model

\\ [E, T, the integer roots of T, sorted] for the minimal model E of ai (for the tests).  The prime table is cleared
\\ first so that long loops stay fast.
tdmodel(ai, P) = removeprimes(addprimes()); addprimes(P); tdmodelE(ellminimalmodel(ellinit(ai)));
tdmodelE(E) = my(T = x^3 + E.b2*x^2 + 8*E.b4*x + 16*E.b6); [E, T, vecsort(Vec(nfroots(, T)))];

\\ [a, b] with T(X + e) = X (X^2 + a X + b), e a root of T
tdab(T, e) =
{
  my(V = Vec(subst(T, x, x + e)));
  if(V[4], error("tdab: ", e, " is not a root of T"));
  [V[2], V[3]];
}

tdisog(ab) = [-2 * ab[1], ab[1]^2 - 4 * ab[2]];

\\ ---------- square classes

\\ the least quadratic non-residue mod the odd prime p
tdeps(p) = my(e = 2); while(kronecker(e, p) != -1, e++); e;

\\ the F_2-basis of Q_v*/Q_v*^2: -1 for R (v = 0); -1, 5, 2 for Q_2; eps, p for odd p
tdbasis(p) = if(p == 0, [-1], p == 2, [-1, 5, 2], [tdeps(p), p]);

\\ the coordinates of the class of d != 0 at v; at p = 2, u = d / 2^v(d) is (-1)^i 5^k mod 8 (1 -> 00, 3 -> 11,
\\ 5 -> 01, 7 -> 10)
tdloc(d, p) =
{
  my(v, u);
  if(p == 0, return([d < 0]));
  v = valuation(d, p); u = d / p^v;
  if(p == 2, u = lift(Mod(u, 8)); return([u % 4 == 3, u == 3 || u == 5, v % 2]));
  [kronecker(lift(Mod(u, p)), p) == -1, v % 2];
}

tdrep(c, p) = my(bs = tdbasis(p)); prod(i = 1, #bs, bs[i]^c[i]);

\\ ---------- local solubility of y^2 = g(u, v), g a binary quartic over Z_p
\\ [MSS96, sec. 6; Cre97, 3.6].  A point of P^1(Q_p) is (s : 1) or (1 : p s) with s in Z_p; each residue class
\\ s = s0 + p s' becomes a new polynomial in s'.  As g is separable, every branch ends: near a point with g != 0 the
\\ value has a fixed valuation and unit class, and near a simple root the polynomial becomes linear with unit slope.

TDENUM = 64;    \\ below this prime, tdsquare tries every residue

\\ 1 if H(s) is a nonzero square mod p for some s (p odd, H mod p != 0).  For p >= TDENUM >= 17 and deg H <= 4,
\\ H = c h^2 q mod p, q squarefree: if deg q >= 1, Weil's bound |sum_s (c q(s) / p)| <= (deg q - 1) sqrt(p) [Lidl and
\\ Niederreiter, Thm. 5.41] leaves more nonzero square values than roots of h; else c must be a square.
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

\\ The children H(r + p x) of one node H, ending in "square" if a square was found.  With c = v_p(content): c >= 2,
\\ H / p^2 has the same squares; c = 1, a square needs H(s)/p = 0 mod p; c = 0, p odd, a unit square mod p lifts
\\ (Hensel); c = 0, p = 2, an odd H(s0) is a square iff 1 mod 8, so only the classes with H(s0) even are refined.
tdzpnode(H, p) =
{
  my(c = valuation(content(H), p), R, L = List());
  if(c >= 2, H /= p^(2 * (c \ 2)); c %= 2);
  if(c == 1, R = tdroots(H / p, p),
    if(p == 2,
      for(s0 = 0, 7, my(h = subst(H, x, s0));
        if(h == 0 || h % 8 == 1, listput(L, "square"); break);
        if(h % 2 == 0, listput(L, subst(H, x, s0 + 8 * x))));
      return(L));
    if(tdsquare(H, p), return(List(["square"])));
    R = tdroots(H, p));
  for(i = 1, #R, listput(L, subst(H, x, R[i] + p * x)));
  L;
}

\\ 1 if H(s) is a square in Q_p (0 included) for some s in Z_p; H nonzero, separable.  An explicit stack, as the depth
\\ grows with the valuations beyond GP's recursion limit.
tdzp(H, p) =
{
  my(N = List([tdzpnode(H, p)]), I = List([0]), k, y);
  while(k = #N,
    if(I[k] == #N[k], listpop(N); listpop(I); next);
    I[k]++; y = N[k][I[k]];
    if(type(y) == "t_STR", return(1));
    listput(N, tdzpnode(y, p)); listput(I, 0));
  0;
}

\\ 1 if y^2 = g(u, v) has a Q_p-point with (u, v) != (0, 0), g = [g4, g3, g2, g1, g0] integral, separable
tdquartic(g, p) =
{
  tdzp(Pol(g), p) || tdzp(subst(Polrev(g), x, p * x), p);
}

\\ 1 if w^2 = d u^4 + A u^2 v^2 + (B/d) v^4 has a point over Q_v (p = 0: R).  Over R with d < 0 it needs
\\ z^2 + A z + B <= 0 for some z <= 0; over Q_p the quartic is scaled by d^2 to be integral.
tdqp(A, B, d, p) =
{
  if(p == 0, return(d > 0 || B < 0 || (A > 0 && A^2 >= 4 * B)));
  tdquartic([d^3, 0, A * d^2, 0, B * d], p);
}

\\ ---------- the local images and the Selmer groups

\\ W_v for (A, B), as coordinate vectors; checked to be a subgroup
tdW(A, B, p) =
{
  my(n = #tdbasis(p), W = List());
  forvec(c = vector(n, i, [0, 1]), if(tdqp(A, B, tdrep(c, p), p), listput(W, c)));
  W = Set(Vec(W));
  foreach(W, u, foreach(W, v, if(!setsearch(W, (u + v) % 2), error("tdW: W_", p, " is not a group"))));
  W;
}

tdimages(A, B, S) = vector(#S + 1, i, my(p = if(i == 1, 0, S[i - 1])); [p, tdW(A, B, p)]);

tdgens(S) = concat([-1], S);

\\ the coordinates of d in Q(S, 2), or the integer 0 if d = 0 or a prime outside S divides d
tdclass(d, S) =
{
  my(c = vector(#S + 1), m = abs(d));
  if(d == 0, return(0));
  c[1] = d < 0;
  for(i = 1, #S, my(v = valuation(m, S[i])); c[i + 1] = v % 2; m /= S[i]^v);
  if(m != 1, 0, c);
}

\\ a basis of Sel(A, B) as signed squarefree integers: d with phi(loc_v(d)) = 0 for every form phi vanishing on W_v
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

\\ 1 if W_v(a, b) and W_v(a', b') are orthogonal complements under the Hilbert symbol at each place (Tate duality)
tddual(Ws, Ws2) =
{
  for(i = 1, #Ws, my(p = Ws[i][1], W = Ws[i][2], W2 = Ws2[i][2]);
    if(#W * #W2 != 2^#tdbasis(p), return(0));
    foreach(W, u, foreach(W2, v, if(hilbert(tdrep(u, p), tdrep(v, p), p) != 1, return(0)))));
  1;
}

\\ [e, a, b, Sel(a, b), Sel(a', b'), s, s', s + s' - 2, duality, b and b' in the Selmer groups] for the root e
tdisogeny(T, e, S) =
{
  my(ab = tdab(T, e), ab2 = tdisog(ab), W = tdimages(ab[1], ab[2], S), W2 = tdimages(ab2[1], ab2[2], S), s1, s2);
  s1 = tdselmer(S, W); s2 = tdselmer(S, W2);
  [e, ab[1], ab[2], s1, s2, #s1, #s2, #s1 + #s2 - 2, tddual(W, W2),
   tdinspan(ab[2], s1, S) && tdinspan(ab2[2], s2, S)];
}

\\ 1 if d is in the span of V in Q(S, 2)
tdinspan(d, V, S) =
{
  my(c = tdclass(d, S), M);
  if(type(c) == "t_INT", return(0));
  if(#V == 0, return(c == 0 * c));
  M = Mat(apply(v -> tdclass(v, S)~, V));
  matrank(Mod(matconcat([M, c~]), 2)) == matrank(Mod(M, 2));
}

\\ 1 if V is independent in Q(S, 2) with the span of V0
tdsamespan(V, V0, S) =
{
  my(C = apply(v -> tdclass(v, S), V), M, M0);
  if(#V != #V0 || #select(c -> type(c) == "t_INT", C), return(0));
  if(#V == 0, return(1));
  M = Mat(apply(c -> c~, C)); M0 = Mat(apply(v -> tdclass(v, S)~, V0));
  matrank(Mod(M, 2)) == #V && matrank(Mod(matconcat([M, M0]), 2)) == #V;
}

\\ ---------- the isogeny pairings Theta_1 (the second step)
\\ The isogeny pairings [Fis17, Thm. 2.1].  C_xi: r^2 = xi s^4 + A s^2 t^2 + (B/xi) t^4 maps to the conic
\\ Q_xi by (X : Y : W) = (s^2 : t^2 : r), and L_xi(s^2, t^2, r) is, up to squares, the pushout function of [Fis17,
\\ sec. 5].  Its value does not depend on P0: two tangent lines differ by a constant times a square on Q_xi (product
\\ formula).  It also bounds r_inf, being at least the 4-descent bound [Fis17, Lemma 2.2], so tdbound's parity step
\\ holds.  Outside V, xi, B/xi, A^2 - 4B and eta are p-units (checked), so C_xi has good reduction; the primitive L_xi
\\ vanishes at <= 4 of its points mod p, while #C_xi(F_p) >= p + 1 - 2 sqrt(p) > 4 for p >= 11 (Hensel lifts the rest).

\\ q(s, t) = xi s^4 + A s^2 t^2 + (B/xi) t^4
tdq(xi, A, B, s, t) = xi * s^4 + A * s^2 * t^2 + B / xi * t^4;

\\ the tangent line alpha X + beta Y + gamma W = 0 (primitive integers) to Q_xi at P0 = [X0, Y0, W0]
tdtangent(xi, A, B, P0) =
{
  my(l = [2 * xi * P0[1] + A * P0[2], A * P0[1] + 2 * B / xi * P0[2], -2 * P0[3]]);
  l = l * denominator(l); l / content(l);
}

\\ The class of L(s^2, t^2, r) at (s, t, r) on C_xi over Q_v, as a rational; 0 if q(s, t) is not a nonzero square,
\\ L = 0 at both roots r, or 100 p-adic digits do not fix the class.  Over R, r > 0; over Q_p either root will do.
tdlclass(xi, A, B, L, s, t, p) =
{
  my(q = tdq(xi, A, B, s, t), u, r, l, v, k);
  if(q == 0, return(0));
  u = L[1] * s^2 + L[2] * t^2;
  if(p == 0,                    \\ the sign of u + L[3] sqrt(q)
    if(q < 0, return(0));
    if(u == 0 || L[3] == 0 || sign(u) == sign(L[3]), return(if(u, sign(u), sign(L[3]))));
    k = sign(u^2 - L[3]^2 * q);
    return(if(k > 0, sign(u), k < 0, sign(L[3]), 0)));
  v = valuation(q, p);
  if(v % 2, return(0));
  if(p == 2, if((q / 2^v) % 8 != 1, return(0)), if(kronecker(lift(Mod(q / p^v, p)), p) != 1, return(0)));
  r = sqrt(q + O(p^(v + 100)));
  l = u + L[3] * r;
  if(l == 0, l = u - L[3] * r);                 \\ L vanishes at (s, t, r): take the point (s, t, -r)
  if(l == 0, return(0));
  if(type(l) != "t_PADIC", return(l));          \\ L[3] = 0: l = u exactly
  v = valuation(l, p); k = padicprec(l, p) - v;
  if(k < if(p == 2, 3, 1), return(0));
  p^v * lift(l / p^v);
}

\\ the matrix of Theta_1 on the basis Xi (rows xi, columns eta), or 0 if a local point is not one
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

\\ Return the rank of Theta_1 on the basis Xi from conic points C and local points Pts, or -1 if a check fails
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

\\ ---------- the Cassels-Tate pairing on Sel_2 (the third step; tdvct, part 2)
\\ dim Sel_2(E) - t = s + s' - 2 - rank Theta_1 [Fis17, Lemma 2.2], so the Cassels-Tate rank is subtracted from that.

\\ [B, S]: B the least of Bs, S the largest integer <= B with (-1)^S = w
tdbound(Bs, w) =
{
  my(B = vecmin(Bs));
  [B, if((B - (w < 0)) % 2, B - 1, B)];
}

\\ Check a 2-isogeny witness and print "TD B S w".  D: per root, [e, a, b, Sel(a, b), Sel(a', b'), s, s', B_e, conic
\\ points, local points and rank of Theta_1, the same for Theta_1', the Cassels-Tate data of E' or 0]; CE: that of E or 0.
tdverify(ai, P, mm, tor, D, CE) =
{
  my(M, E, S, R, w, ok, BS, got = List(), rkE = -1);
  removeprimes(addprimes());          \\ the prime table holds only P
  E = vcurve(ai, P, mm); M = tdmodelE(E);
  vtors(E[1..5], tor);
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

\\ ==================== 4. curves with E(Q)[2] = 0: formats 3 and 2, and 4 to 6

\\ [BDF08, Cor. 2.2, (10)]: under GRH, the primes of norm < T generate Cl(K) if
\\   2 sum_{p, m: Np^m < T} log Np / Np^(m/2) * (1 - log Np^m / log T)
\\       > log |d_K| - n (gamma + log 8 pi - c1 / log T) - r1 (pi/2 - c2 / log T),   c1 = pi^2/2, c2 = 4 Catalan.
\\ (PARI's GRHchk makes the same test.)  It need not be monotone in T, so it is checked at the T used.

\\ splitting data [p, log p, [f_1, ..]] for the primes p < C
bdfdata(nf, C) =
{
  my(L = List()); forprime(p = 2, C - 1, listput(L, [p, log(p), apply(q -> q.f, idealprimedec(nf, p))])); Vec(L);
}

\\ (10) in exact interval arithmetic: each real becomes rationals lo <= x <= hi with denominator 2^BDFK.
BDFK = 128;
\\ pi, Euler, Catalan, log pi, each truncated to 40 digits: c in [C, C + 10^-40]
bdfconst() =
{
  [31415926535897932384626433832795028841971, 5772156649015328606065120900824024310421,
   9159655941772190150546035149323841107741, 11447298858494001741434273513530587116472] / 10^40;
}

\\ [lo, hi] around 2 atanh z, 0 <= z <= 1/3: the series' tail is below 2 t / (1 - z^2) <= 9/4 t, t the first term
\\ left out (without its 1/(2j+1))
bdfat(z) =
{
  my(s = 0, t = z, j = 0, S = 2^BDFK);
  while(t > 2^(-BDFK - 4), s += t / (2 * j + 1); t *= z^2; j++);
  [floor(2 * s * S), ceil((2 * s + 9 / 4 * t) * S)] / S;
}

\\ [lo, hi] around log n = m log 2 + 2 atanh((n - 2^m) / (n + 2^m)), 2^m <= n < 2^(m+1); l2 = bdfat(1/3) is log 2
bdflog(n, l2) = my(m = logint(n, 2)); m * l2 + bdfat((n - 2^m) / (n + 2^m));

\\ 1 if (10) holds at T, proved in the form (all terms positive)
\\   log T log |d_K| + cN + 2 SB < log T (n (gamma + 3 log 2 + log pi) + r1 pi / 2) + 2 SA log T,
\\ with upper bounds on the left and lower bounds on the right
bdfcert(nf, D, T) =
{
  my(N = poldegree(nf.pol), R1 = nf.sign[1], S = 2^BDFK, C = bdfconst(), e = 10^-40, l2 = bdfat(1/3),
     lT = bdflog(T, l2), ld = bdflog(abs(nf.disc), l2), SA = 0, SB = 0, L, R);
  if(#D && D[#D][1] < T - 1 && nextprime(D[#D][1] + 1) < T, error("bdfcert: splitting data stops below T"));
  for(i = 1, #D, my(p = D[i][1], lp); if(p >= T, break); lp = bdflog(p, l2);
    foreach(D[i][3], f, my(Np = p^f, X = Np, m = 0, s = 0, w = 0, q);
      \\ Np^(-m/2) in [q, q + 1] / S, as sqrt(4^BDFK / Np^m) lies in [sqrtint(4^BDFK \ Np^m), that + 1)
      while(X < T, m++; q = sqrtint(S^2 \ X); s += q; w += m * (q + 1); X *= Np);
      SA += f * lp[1] * s / S; SB += f^2 * lp[2]^2 * w / S));
  L = lT[2] * ld[2] + N * (C[1] + e)^2 / 2 + 4 * R1 * (C[3] + e) + 2 * SB;
  R = lT[1] * (N * (C[2] + 3 * l2[1] + C[4]) + R1 * C[1] / 2) + 2 * SA * lT[1];
  L < R;
}

\\ The general relations (check.c does a - b alpha and (p)): print "R i" and the odd-valuation primes, named as in
\\ check.c.  P lists the primes of disc(f).

ckinit(f, P) =
{
  my(nf, al);
  if(poldegree(f) != 3 || !polisirreducible(f), error("F(x, 1) is not an irreducible cubic"));
  addprimes(P); [nf, al] = nfinit([f, concat([2, 3], P)], 3);
  if(#nfcertify(nf), error("the maximal order is not certified"));
  if(subst(f, x, al) != 0, error("ckinit: alpha is not a root of f"));
  [nf, al, factor(abs(pollead(f)))[, 1]~, f];
}
ckdec(D, nf, p) = my(d); if(mapisdefined(D, p, &d), d, d = idealprimedec(nf, p); mapput(D, p, d); d);
\\ the name of P above p: "p:r", r the point alpha reduces to (r = p when v_P(alpha) < 0), or "p::f" for degree
\\ f > 1; unique as R_F = O_K
ckcol(K, p, P) =
{
  if(P.f > 1, return(Str(p, "::", P.f)));
  if(idealval(K[1], K[2], P) < 0, return(Str(p, ":", p)));
  Str(p, ":", polcoef(nfmodpr(K[1], K[2], nfmodprinit(K[1], P)).pol, 0));
}

\\ rels: general relations [c0, c1, c2, d, H]; I: their indices; errors on a failed check
ckrels(K, f, rels, I) =
{
  my(D = [Map(), Map(), Map()]);        \\ idealprimedec, ckpts and isprime, by p
  for(i = 1, #rels, if(#rels[i] != 5, error("relation ", I[i], " is not a general relation"));
    print("R ", I[i], " ", strjoin(Vec(ckgen(K, D, rels[i], I[i])), " ")));
  #rels;
}

\\ The odd-valuation columns of x = (c0 + c1 alpha + c2 alpha^2)/d.  Away from lc(f) d, x is integral, so
\\ sum f_P v_P(x) = v_p(N(x)) with v_P >= 0: only the primes of lc(f), d and N(x) need examining.  That N(x) is
\\ supported on H, lc(f) and d is checked (by gcds with d) before d is factored and the primes proved.
ckgen(K, D, r, i) =
{
  my(nf = K[1], al = K[2], d = r[4], H = r[5], el, n, ps, m, g, dd, cols = List(), Jl = Set(K[3]),
     HL = Set(concat(H, K[3])));
  if(d <= 0, error("relation ", i, " has d <= 0"));
  el = (r[1] + r[2] * al + r[3] * al^2) / d;
  if(el == 0, error("relation ", i, " is zero"));
  n = nfeltnorm(nf, el);
  foreach(HL, p, if(p < 2, error("relation ", i, " has ", p, ", which is not prime")));
  m = numerator(abs(n)) * denominator(n); foreach(HL, p, m /= p^valuation(m, p));
  while((g = gcd(m, d)) > 1, m /= g);
  if(m != 1, error("relation ", i, " does not factor as claimed: a prime of its norm is not given"));
  dd = d; foreach(HL, p, dd /= p^valuation(dd, p));
  ps = Set(concat([H, K[3], if(dd > 1, factor(dd)[, 1]~, [])]));
  foreach(ps, p, if(!ckprime(D[3], p), error("relation ", i, " has ", p, ", which is not prime")));
  foreach(ps, p, my(dec, full = (!setsearch(Jl, p) && d % p != 0), vp = if(full, valuation(n, p), 0), s = 0);
    if(full && vp == 0, next);
    \\ v_p(N(x)) = 1, x integral at p: one P, of degree 1, has v_P(x) = 1, the one with c0 + c1 r + c2 r^2 = 0 mod p
    if(full && vp == 1,
      my(hit = select(z -> (r[1] + r[2] * z[2] + r[3] * z[2]^2) % p == 0, ckpts(D[2], K, p)));
      if(#hit != 1, error("relation ", i, " has ", #hit, " degree-1 primes above ", p, " with v_p(N(x)) = 1"));
      listput(cols, hit[1][1]); next);
    dec = ckdec(D[1], nf, p);
    for(k = 1, #dec, my(v = idealval(nf, el, dec[k]));
      if(v % 2, listput(cols, ckcol(K, p, dec[k])));
      if(full, s += v * dec[k].f; if(s == vp, break))));
  cols;
}
ckprime(M, p) = my(b); if(mapisdefined(M, p, &b), b, b = isprime(p); mapput(M, p, b); b);
\\ [[name, r], ...] for the roots r of f mod p (p prime to lc(f)): as R_F = O_K, the degree-1 primes above p
ckpts(M, K, p) =
{
  my(t);
  if(mapisdefined(M, p, &t), return(t));
  t = [[Str(p, ":", z), z] | z <- apply(lift, polrootsmod(K[4], p))];
  mapput(M, p, t); t;
}

\\ the number of degree-1 primes of norm < GB
ckns(K, GB) = my(ns = 0); forprime(p = 2, GB - 1, foreach(idealprimedec(K[1], p), P, ns += (P.f == 1))); ns;

\\ Brumer-Kramer terms [u, n] [KSW19, Prop. 3.1]: E minimal, nf its 2-division field
vbk(E, nf, P) =
{
  my(D = E.disc, n = 0);
  foreach(P, p, my(red = elllocalred(E, p));
    if(red[1] == 1, n += (valuation(D, p) % 2 == 0), red[1] >= 2, n += #idealprimedec(nf, p) - 1));
  [if(D > 0, 2, 1), n];
}

\\ Check O_K and d_K of nf = Q[x]/(T), that F is an index form (disc F = d_K, so R_F = O_K) and Q(alpha) = K.
\\ Return [nf, ckinit(f, P)].
vorder(T, P, f, dKc) =
{
  my(nf, K, ok);
  if(!polisirreducible(T), error("W is reducible, so E(Q)[2] != 0"));
  nf = nfinit([T, concat([2, 3], P)]);
  \\ a wrong d_K or F stops here, before the work it would bound or allow (B, ns, ckinit, check rel)
  ok = #nfcertify(nf) == 0 && nf.disc == dKc;
  vchk("maximal_order", ok, "d_K");
  if(!ok, vhalt());
  ok = poldegree(f) == 3 && poldisc(f) == dKc && content(f) == 1;
  vchk("index_form", ok, "disc F = d_K, so R_F = O_K");
  if(!ok, vhalt());
  K = ckinit(f, P);
  vchk("same_field", nfisisom(K[1], nf) != 0);
  [nf, K];
}

\\ the field-level checks of format 3
vfield(ai, P, mm, f, dKc, GBc, nsc, uc, nc, wc) =
{
  my(E, T, nf, K, u, n, ns);
  E = vcurve(ai, P, mm);
  T = x^3 + E.b2*x^2 + 8*E.b4*x + 16*E.b6;
  vchk("E2_trivial", polisirreducible(T), "2-division cubic irreducible: E(Q)[2] = 0");
  [nf, K] = vorder(T, P, f, dKc);
  [u, n] = vbk(E, nf, P);
  vchk("bk_u", u == uc, Str("u = ", u));
  vchk("bk_n", n == nc, Str("n = ", n));
  vrootno(E, wc);
  vchk("bdf_GB", GBc > 1 && bdfcert(K[1], bdfdata(K[1], GBc), GBc),
       Str("BDF (10) holds at T = ", GBc, " (exact interval arithmetic)"));
  ns = ckns(K, GBc); vchk("ns", ns == nsc, Str("degree-1 primes of norm < GB: ", ns));
  K;
}

\\ ---------- formats 4 to 6
\\ K = ckinit(f, P) = [nf, al, primes of lc(f)]; W = x^3 + b2 x^2 + 8 b4 x + 16 b6, W(theta) = 0.  Data lines:
\\   NUM name value      r1, r2, nSK (#S_K), u, n, w
\\   BLK p size d        place p (0: the real places): its block of class coordinates, d = dim E(Q_p)/2E(Q_p)
\\   PI p l bits         the class of x_l - theta at p
\\   SC t name           the column name of s_columns[t]
\\   RC j bits           relation j: its blocks in BLK order, the real signs, the norm's sign and valuations mod 2 at
\\                       S, then its characters
\\   CTR rk              the rank of the Cassels-Tate pairing, or -1

\\ ---------- exact signs at the real places

\\ a rational interval isolating the real root rt of f, f(l) f(u) < 0; v4isos lists the roots in increasing order,
\\ the order of the witness's real character indices
v4iso(f, rt) =
{
  my(a = floor(rt * 2^200) / 2^200, l = a - 1 / 2^180, u = a + 1 / 2^180);
  if(polsturm(f, [l, u]) != 1 || sign(subst(f, x, l)) * sign(subst(f, x, u)) >= 0, error("v4iso: root not isolated"));
  [l, u];
}
v4isos(f) = { my(R); localbitprec(800); R = polrootsreal(f); vector(#R, k, v4iso(f, R[k])); }

\\ the sign of g(alpha), deg g < 3, alpha the root of f in iv: bisect until g has no root there (g(alpha) != 0, as
\\ f is an irreducible cubic)
v4sign(f, iv, g) =
{
  my(l = iv[1], u = iv[2], sl = sign(subst(f, x, l)));
  if(g == 0, error("v4sign: zero"));
  if(poldegree(g) <= 0, return(sign(polcoef(g, 0))));
  while(polsturm(g, [l, u]) > 0, my(m = (l + u) / 2, sm = sign(subst(f, x, m)));
    if(sm == 0, error("v4sign: rational root of f"));
    if(sm == sl, l = m, u = m));
  sign(subst(g, x, l));
}

\\ ---------- the places of K above S: classes in K_P*/K_P*^2

\\ [p, pr, D, even indices of bid.cyc, s]: the class of y at pr is (v_pr(y) mod 2, the unit part y tau^v in
\\ (O/pr^n)*/squares), tau PARI's anti-uniformizer.  Above 2, n = 2e + 1 and ideallog reads the components of even
\\ order; above odd p, n = 1 and Euler's criterion gives the bit without a discrete log.  s = p tau^e is a unit at pr
\\ and divisible by the other Q | p, so a factor s^2m clears denominators there without changing the class.
v4place(nf, p, pr) =
{
  my(D, ev, s);
  if(p == 2,
    D = idealstar(nf, idealpow(nf, pr, 2 * (pr.e) + 1), 1); ev = select(d -> d % 2 == 0, D.cyc, 1);
    if(#ev != 1 + (pr.e) * (pr.f), error("v4place: (O/P^n)*/squares has the wrong dimension at ", p)),
    D = nfmodprinit(nf, pr); ev = [1]);
  nfeltval(nf, p, pr, &s);
  if(idealval(nf, s, pr) != 0, error("v4place: s"));
  [p, pr, D, ev, s];
}
v4places(nf, SP) = my(L = List()); foreach(SP, p, foreach(idealprimedec(nf, p), pr, listput(L, v4place(nf, p, pr)))); Vec(L);

\\ the class of y != 0 at the place pl
v4cls(nf, y, pl) =
{
  my(pr = pl[2], z, v = nfeltval(nf, y, pr, &z), dec = idealprimedec(nf, pl[1]), m = 0, lg, e);
  foreach(dec, Q, if(Q != pr, my(w = idealval(nf, z, Q)); if(w < 0, m = max(m, ceil(-w / (2 * (Q.e)))))));
  if(m, z = nfeltmul(nf, z, nfeltpow(nf, pl[5], 2 * m)));
  if(idealval(nf, z, pr) != 0 || denominator(nfalgtobasis(nf, z)) % pl[1] == 0, error("v4cls: unit part"));
  if(pl[1] == 2, lg = ideallog(nf, z, pl[3]); return(concat([v % 2], vector(#pl[4], i, lg[pl[4][i]] % 2))));
  e = nfmodpr(nf, z, pl[3])^((pl[1]^(pr.f) - 1) / 2);
  if(e != 1 && e != -1, error("v4cls: Euler's criterion"));
  [v % 2, e != 1];
}

\\ ---------- the relations as elements of K

\\ [the relation r as a polynomial in x = alpha, its norm]
v4elt(r, F2, lc, nf, al) =
{
  if(#r == 1, return([r[1], r[1]^3]));
  if(#r == 3, return([r[1] - r[2] * x, subst(subst(F2, x, r[1]), y, r[2]) / lc]));
  my(g = (r[1] + r[2] * x + r[3] * x^2) / r[4]);
  [g, nfeltnorm(nf, subst(g, x, al))];
}

\\ the character of r at [q, c]: the Legendre symbol of its image under alpha -> c, as 0 or 1
v4chi(r, q, c, n) =
{
  my(t);
  if(valuation(n, q) != 0, error("the character ", q, " divides the norm of a relation used"));
  if(#r == 1, t = r[1],
     #r == 3, t = r[1] - r[2] * c,
     if(r[4] % q == 0, error("the character ", q, " divides a denominator")); t = (r[1] + r[2] * c + r[3] * c^2) * r[4]);
  t = kronecker(t % q, q);
  if(t == 0, error("the character ", q, ": the relation is not a unit at the prime"));
  (1 - t) / 2;
}

\\ ---------- the checks

vbits(v) = concat(apply(z -> Str(z), Vec(v)));

\\ The field-level checks of format 4; bt is 1 (BDF) or 2 (Minkowski).  Return [K, W, theta in K, E, theta as a
\\ polynomial in alpha].
v4field(ai, P, mm, f, dKc, Wc, th, bt, B, nsc, wc, SP, DR) =
{
  my(E, T, nf, K, ns, r1, C, al, tt, nSK = 0, Em, ok = 1);
  E = vcurve(ai, P, mm);
  \\ a bad p with odd c_p may be left out of S [SS04]
  Em = ellinit(mm);
  foreach(DR, z, ok = ok && z[1] % 2 && setsearch(Set(P), z[1]) && elllocalred(Em, z[1])[4] == z[2] && z[2] % 2);
  if(#DR, vchk("s_dropped", ok, Str(#DR, " dropped primes: odd, bad, c_p as claimed and odd")));
  T = x^3 + E.b2*x^2 + 8*E.b4*x + 16*E.b6;
  vchk("E2_trivial", polisirreducible(T), "2-division cubic irreducible: E(Q)[2] = 0");
  vchk("weierstrass", Pol(Vecrev(Wc)) == T, "W = x^3 + b2 x^2 + 8 b4 x + 16 b6 of the minimal model");
  [nf, K] = vorder(T, P, f, dKc); al = K[2];
  tt = th[1] + th[2] * al + th[3] * al^2;
  vchk("theta", subst(T, x, tt) == 0, "W(theta) = 0 in K, exactly");
  r1 = K[1].sign[1];
  print("NUM r1 ", r1); print("NUM r2 ", K[1].sign[2]);
  foreach(SP, p, nSK += #idealprimedec(K[1], p)); print("NUM nSK ", nSK);
  my(un = vbk(E, nf, P)); print("NUM u ", un[1]); print("NUM n ", un[2]);
  my(w = vrootno(E, wc)); print("NUM w ", w);
  if(bt == 1,
    vchk("bound", B > 1 && bdfcert(K[1], bdfdata(K[1], B), B), Str("BDF (10) holds at T = ", B, " (exact interval arithmetic)")),
    C = bdfconst()[1];
    \\ M = (4/pi)^r2 (2/9) sqrt|d_K| < B: M^2 <= (4/C)^(2 r2) (4/81) |d_K| < B^2 (pi >= C)
    vchk("bound", B > 1 && (4 / C)^(2 * K[1].sign[2]) * 4 / 81 * abs(dKc) < B^2, Str("Minkowski bound < B = ", B, " (exact)")));
  ns = ckns(K, B); vchk("ns", ns == nsc, Str("degree-1 primes of norm < B: ", ns));
  [K, T, tt, E, th[1] + th[2] * x + th[3] * x^2];
}

\\ the column name of [p, r]: the degree-1 P with v_P((r - alpha) J) > 0, or v_P(J) > 0 for r = p; J^-1 = O_K + alpha O_K
v4label(K, f, p, r) =
{
  my(nf = K[1], al = K[2], dec = idealprimedec(nf, p), Ji = idealhnf(nf, 1, al), el, hit = List());
  if(!isprime(p) || r < 0 || r > p, error("s_column [", p, ", ", r, "] is not a label"));
  if(r < p, if(subst(f, x, r) % p, error("s_column [", p, ", ", r, "]: r is not a root mod p")); el = r - al,
     if(pollead(f) % p, error("s_column [", p, ", ", p, "]: (1:0) needs p | lc(F)")); el = 1);
  for(k = 1, #dec, if(dec[k].f == 1 && idealval(nf, el, dec[k]) - idealval(nf, Ji, dec[k]) > 0, listput(hit, k)));
  if(#hit != 1, error("s_column [", p, ", ", r, "]: ", #hit, " primes"));
  ckcol(K, p, dec[hit[1]]);
}

\\ the root counts and the points at each p of SP and at the real places, checked, with their classes
v4local(K, f, T, tt, tp, SP, X, Xinf, pls, iso) =
{
  my(nf = K[1], al = K[2], ok, d, nr, nr2, g);
  for(i = 1, #SP, my(p = SP[i], blk = select(z -> z[1] == p, pls), sz = 0);
    nr = #select(pr -> (pr.e) * (pr.f) == 1, idealprimedec(nf, p));
    nr2 = #polrootspadic(T, p, valuation(poldisc(T), p) + 10);
    vchk(Str("roots_", p), nr == nr2, Str(nr, " roots of W in Q_", p, " (idealprimedec, polrootspadic)"));
    d = logint(nr + 1, 2) + (p == 2);            \\ dim E(Q_p)[2] + [p = 2]: #roots 0, 1, 3 give dim 0, 1, 2
    foreach(blk, pl, sz += 1 + #pl[4]);
    print("BLK ", p, " ", sz, " ", d);
    ok = 1;
    for(l = 1, #X[i], my(xl = X[i][l], Wx = subst(T, x, xl), v, u, t);
      if(Wx == 0, ok = 0; break);
      v = valuation(Wx, p); u = Wx / p^v; t = numerator(u) * denominator(u);
      if(v % 2 || (p == 2 && t % 8 != 1) || (p > 2 && kronecker(t, p) != 1), ok = 0; break);
      print("PI ", p, " ", l - 1, " ", vbits(concat(apply(pl -> v4cls(nf, xl - tt, pl), blk)))));
    vchk(Str("points_", p), ok, Str(#X[i], " points, W(x) a nonzero square in Q_", p)));
  nr = polsturm(T);
  vchk("roots_inf", nr == #iso, Str(nr, " real roots of W"));
  print("BLK 0 ", #iso, " ", logint(nr + 1, 2) - 1);   \\ dim E(R)[2] - 1: 0 or 1
  ok = 1;
  for(l = 1, #Xinf, my(xl = Xinf[l]);
    if(subst(T, x, xl) <= 0, ok = 0; break);
    g = xl - tp;
    print("PI 0 ", l - 1, " ", vbits(vector(#iso, k, v4sign(f, iso[k], g) < 0))));
  vchk("points_inf", ok, Str(#Xinf, " points, W(x) > 0"));
}

\\ the RC lines of the relations R (indices J); RE: real embeddings, 0-based; CQ: [[q, c], ...]
v4rels(K, f, SP, R, J, pls, iso, RE, CQ) =
{
  my(nf = K[1], al = K[2], F2 = substvec(f, [x], [x / y]) * y^3, lc = pollead(f));
  foreach(CQ, z, my(q = z[1], c = z[2]);
    if(!isprime(q) || setsearch(SP, q) || lc % q == 0 || c < 0 || c >= q || subst(f, x, c) % q,
      error("the character [", q, ", ", c, "] is not one: q must be a prime outside S, prime to lc(F), c a root of F(x, 1) mod q")));
  foreach(RE, k, if(k < 0 || k >= #iso, error("the character ", k, " is no real embedding")));
  for(j = 1, #R, my(r = R[j], en = v4elt(r, F2, lc, nf, al), g = en[1], n = en[2], el = subst(g, x, al), b = List());
    if(n == 0, error("relation ", J[j], " is zero"));
    foreach(pls, pl, listput(b, v4cls(nf, el, pl)));
    listput(b, vector(#iso, k, v4sign(f, iso[k], g) < 0));
    listput(b, concat([n < 0], vector(#SP, i, valuation(n, SP[i]) % 2)));
    listput(b, vector(#RE, k, v4sign(f, iso[RE[k] + 1], g) < 0));
    listput(b, vector(#CQ, k, v4chi(r, CQ[k][1], CQ[k][2], n)));
    print("RC ", J[j], " ", vbits(concat(Vec(b)))));
}

\\ the RC lines alone, for the other gp processes of a split check (the first makes the checks)
v4relsonly(f, P, SP, R, J, RE, CQ) =
{
  my(K = ckinit(f, P));
  v4rels(K, f, SP, R, J, v4places(K[1], SP), v4isos(f), RE, CQ);
}

v4main(KT, f, SP, SCOL, X, Xinf, R, J, RE, CQ) =
{
  my(K = KT[1], pls = v4places(K[1], SP), iso = v4isos(f));
  for(t = 1, #SCOL, print("SC ", t - 1, " ", v4label(K, f, SCOL[t][1], SCOL[t][2])));
  v4local(K, f, KT[2], KT[3], KT[5], SP, X, Xinf, pls, iso);
  v4rels(K, f, SP, R, J, pls, iso, RE, CQ);
}

\\ the Cassels-Tate section of formats 4 to 6
v4ct(mm, P, CT) =
{
  my(rk = tdvct(mm, Set(concat([2], P)), CT[1], CT[2], CT[3]));
  vchk("cassels_tate", rk >= 0 && rk == CT[4], Str("rank of the Cassels-Tate pairing on the quartics: ", rk));
  print("CTR ", rk);
}
