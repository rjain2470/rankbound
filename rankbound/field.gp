\\ Field data for the prover (loads verify.gp, for bdf and vbk).  E has E(Q)[2] = 0 and K = Q[x]/(T), T the 2-division
\\ cubic.  F is the index form of O_K (disc F = d_K), GL2(Z)-reduced with leading coefficient > 0, and alpha a root of
\\ F(x, 1); a relation (a, b) is the element a - b alpha, of norm F(a, b)/lc(F).
\\ A degree-1 prime P above p is labeled by the point (r:1) or (1:0) of P^1(F_p) with P | (a - b alpha) J, where
\\ J^-1 = O_K + alpha O_K, and r = p stands for (1:0), as in CADO-NFS.  For p not dividing lc(F) this is
\\ Kummer-Dedekind, P = (p, alpha - r); for p | lc(F) the label is checked to be unique.

\\ GL2(Z)-reduction of the binary cubic F(x, y) (weights w on |x - theta_i y|^2)
sfred(F, w) =
{
  my(th = polroots(subst(F, y, 1)), G = matrix(2, 2), U);
  for(i = 1, 3, my(t = th[i]); G += w[i] * [1, -real(t); -real(t), abs(t)^2]);
  U = qflllgram(G); substvec(F, [x, y], [U[1,1]*x + U[1,2]*y, U[2,1]*x + U[2,2]*y]);
}
\\ [log2 of the skewed size max_i |c_i| s^(i - 3/2), skew s] (s a power of 2^(1/4))
sfsize(f) =
{
  my(best = oo, sk = 1);
  for(k = -400, 400, my(s = 2^(k / 4.), m = 0);
    for(i = 0, 3, my(ci = polcoef(f, i)); if(ci, m = max(m, abs(ci) * s^(i - 1.5))));
    if(m < best, best = m; sk = s));
  [log(best) / log(2), sk];
}
\\ the real roots of f, precise enough to round
realroots(f) = localprec(38 + #Str(vecmax(abs(Vec(f))))); polrootsreal(f);
\\ labels of the degree-1 primes above p: [[r, k, e], ...], k = index in idealprimedec, e = ramification index
sflabels(nf, al, Jinv, f, p) =
{
  my(dec = idealprimedec(nf, p), lc = pollead(f), rts = List(), out = List(), n1 = 0);
  foreach(polrootsmod(f, p), r, listput(rts, [lift(r), lift(r), 1]));
  if(lc % p == 0, listput(rts, [p, 1, 0]));
  for(k = 1, #dec, n1 += dec[k].f == 1);
  if(#rts != n1, error("sflabels: ", #rts, " roots but ", n1, " degree-1 primes above ", p));
  foreach(Vec(rts), t, my(hit = List(), el = t[2] - t[3] * al);
    for(k = 1, #dec, if(dec[k].f == 1 && idealval(nf, el, dec[k]) - idealval(nf, Jinv, dec[k]) > 0, listput(hit, k)));
    if(#hit != 1, error("sflabels: root ", t[1], " mod ", p, " hits ", #hit, " primes"));
    listput(out, [t[1], hit[1], dec[hit[1]].e]));
  Vec(out);
}
\\ Brumer-Kramer terms by the verifier's vbk: [u, n, [[p, f_p, v_p(Delta), p's term of n], ...]], with f_p the
\\ conductor exponent
sfbk(E, nf, P) =
{
  my(un = vbk(E, nf, P));
  [un[1], un[2], vector(#P, i, my(p = P[i]); [p, elllocalred(E, p)[1], valuation(E.disc, p), vbk(E, nf, [p])[2]])];
}
\\ ai: the a-invariants; P: every prime of the minimal discriminant (checked here)
sfield(ai, P) =
{
  my(E, D, T, nf, dK, Bm, Bi, g, g2, co, Iform, F, th, dT, cands, ff, ss, rr, ks, GB, bach, nfF, al, Jinv, bk, w, ns = 0,
     lc, dKp, Jv = List());
  foreach(P, p, if(p < 2, error("The bad prime ", p, " is not prime.")));
  foreach(P, p, if(!ispseudoprime(p), error("The bad prime ", p, " is not prime.")));
  if(#Set(P) != #P, error("The bad prime ", select(p -> #select(q -> q == p, P) > 1, P)[1], " is listed twice."));
  addprimes(P); E = ellminimalmodel(ellinit(ai)); D = E.disc;
  foreach(P, p, if(D % p, error("The bad prime ", p, " does not divide the discriminant.")));
  if(abs(D) != prod(i = 1, #P, P[i]^valuation(D, P[i])), error("The bad primes given do not account for the whole discriminant."));
  T = x^3 + E.b2*x^2 + 8*E.b4*x + 16*E.b6;
  if(!polisirreducible(T), error("E2 the 2-division cubic is reducible"));
  nf = nfinit([T, concat([2, 3], P)]); dK = nf.disc;
  if(#nfcertify(nf), error("sfield: maximal order not certified"));
  \\ index form of O_K (Delone-Faddeev): F(u, v) = det of the coordinates of 1, g, g^2 with g = u w2 + v w3
  Bm = matrix(3, 3, i, k, polcoef(nf.zk[k], i - 1)); Bi = Bm^-1;
  g = 'u * nf.zk[2] + 'v * nf.zk[3]; g2 = lift(Mod(g, nf.pol)^2);
  co = (z -> Bi * vector(3, i, polcoef(z, i - 1, x))~);
  Iform = substvec(matdet(matconcat([co(1), co(g), co(g2)])), ['u, 'v], [x, y]);
  if(poldisc(subst(Iform, y, 1)) != dK, error("sfield: index form disc != d_K"));
  th = polroots(subst(Iform, y, 1)); dT = deriv(subst(Iform, y, 1));
  cands = [sfred(Iform, [1, 1, 1]), sfred(Iform, vector(3, i, abs(subst(dT, x, th[i]))^(-2/3)))];
  cands = apply(h -> if(pollead(subst(h, y, 1)) < 0, -subst(h, y, 1), subst(h, y, 1)), cands);
  ff = if(sfsize(cands[1])[1] <= sfsize(cands[2])[1], cands[1], cands[2]);
  if(poldisc(ff) != dK || poldegree(ff) != 3, error("sfield: reduced form"));
  ss = sfsize(ff); rr = realroots(ff); ks = vector(#rr, i, round(rr[i]));
  bach = floor(12 * log(abs(dK))^2);
  GB = bdf(nf, bach);
  \\ the field of F: alpha a root of ff, R_F = O_K since disc F = d_K
  [nfF, al] = nfinit([ff, concat([2, 3], P)], 3);
  if(nfF.disc != dK || !nfisisom(nfF, nf), error("sfield: field of F"));
  lc = pollead(ff); Jinv = idealhnf(nfF, 1, al);
  if(idealnorm(nfF, Jinv) != 1 / lc, error("sfield: N(O + alpha O) != 1/lc"));
  bk = sfbk(E, nf, P); w = ellrootno(E);
  print("SF\tainvs\t", strjoin(apply(z -> Str(z), Vec(E[1..5])), ","));
  print("SF\tpoly\t", strjoin(apply(z -> Str(z), Vecrev(Vec(ff))), ","));
  print("SF\tdK\t", dK);
  print("SF\tlog2dK\t", log(abs(dK)) / log(2));
  print("SF\tr1\t", nf.sign[1]);
  print("SF\tGB\t", GB);
  print("SF\tbach\t", bach);
  print("SF\tlog2size\t", ss[1]);
  print("SF\tlog2skew\t", log(ss[2]) / log(2));
  print("SF\treal_roots\t", strjoin(apply(z -> Str(z), ks), ","));
  print("SF\tu\t", bk[1]);
  print("SF\tn\t", bk[2]);
  print("SF\tbk_primes\t", strjoin(apply(z -> Str(z[1], ":", z[2], ":", z[3], ":", z[4]), bk[3]), ","));
  print("SF\tw\t", w);
  print("SF\tsignD\t", sign(E.disc));
  \\ J: odd valuations, as labels where the prime has degree 1 (else p:k:f, a large column)
  foreach(factor(lc)[, 1]~, p, my(lab = sflabels(nfF, al, Jinv, ff, p), dec = idealprimedec(nfF, p));
    for(k = 1, #dec, my(v = -idealval(nfF, Jinv, dec[k]));
      if(v % 2, my(t = select(z -> z[2] == k, lab)); listput(Jv, if(#t, Str(p, ":", t[1][1]), Str(p, ":k", k, ":f", dec[k].f))))));
  print("SF\tJ\t", strjoin(Vec(Jv), ","));
  \\ ramified primes: label:e for the degree-1 primes above p, and whether all primes above p have degree 1
  dKp = factor(abs(dK))[, 1]~;
  foreach(dKp, p, my(lab = sflabels(nfF, al, Jinv, ff, p), dec = idealprimedec(nfF, p));
    print("RAM\t", p, "\t", vecsum(apply(q -> q.f, dec)) == #dec, "\t", strjoin(apply(z -> Str(z[1], ":", z[3]), lab), ",")));
  \\ the small columns: every degree-1 prime of norm < GB
  forprime(p = 2, GB - 1, my(lab = sflabels(nfF, al, Jinv, ff, p));
    foreach(lab, z, ns++; print("S\t", p, "\t", z[1])));
  print("SF\tns\t", ns);
}

\\ the primes p < pmax, then those of X, at which F has three roots in P^1(F_p), as makefb's lines "p: r1,r2,r3"
sfsplit1(f, c, p) =
{
  my(rs);
  if(content(f) % p,
    rs = apply(z -> lift(z), Vec(polrootsmod(f, p)));
    if(c[4] % p == 0, rs = concat(rs, [p]));
    if(#rs == 3, print(p, ": ", strjoin(apply(z -> Str(z), rs), ","))));
}
sfsplit(c, pmax, X = []) =
{
  my(f = Pol(Vecrev(c)));
  forprime(p = 2, pmax - 1, sfsplit1(f, c, p));
  foreach(X, p, sfsplit1(f, c, p));
}

\\ the points PTS of the model a0 as points of the model a, which must have the same minimal model; each point is
\\ checked on both models (P: the bad primes)
rkmap(a0, a, P, PTS) =
{
  my(E0, E, M0, M, v0, v);
  addprimes(P); E0 = ellinit(a0); E = ellinit(a);
  M0 = ellminimalmodel(E0, &v0); M = ellminimalmodel(E, &v);
  if(M0[1..5] != M[1..5], error("The points are not on this curve."));
  vector(#PTS, i, my(Q);
    if(!ellisoncurve(E0, PTS[i]), error("Point ", i, " is not on the curve."));
    Q = ellchangepointinv(ellchangepoint(PTS[i], v0), v);
    if(!ellisoncurve(E, Q), error("rkmap: point ", i, " is not on the certificate's model"));
    Q);
}

\\ the first r primes, from PR and then below pmax, at which the 2-signatures of the points PTS have rank r
rksel(ai, r, PTS, PR, pmax) =
{
  my(E0 = ellinit(ai), rows = List(), keep = List(), rk = 0);
  if(r == 0, return([]));
  PR = concat(PR, setminus(Set(primes([3, pmax - 1])), Set(PR)));
  for(k = 1, #PR, my(p = PR[k], Ep, n, h, row, M);
    if(!isprime(p) || p < 3 || E0.disc % p == 0, next);
    Ep = ellinit(ai, p); n = ellcard(Ep); if(n % 4 != 2, next); h = n / 2;
    row = vector(r, i, if(denominator(PTS[i][1]) % p == 0 || denominator(PTS[i][2]) % p == 0, 0,
      if(ellmul(Ep, [Mod(PTS[i][1], p), Mod(PTS[i][2], p)], h) == [0], 0, 1)));
    M = matconcat(concat(Vec(rows), [row])~);
    if(matrank(Mod(M, 2)) > rk, listput(rows, row); listput(keep, p); rk++);
    if(rk == r, break));
  Vec(keep);
}

\\ the rank lower bound on the model a: prints "PT x y" per mapped point and "PR p1,p2,..."
rkproof(a0, a, P, r, PTS, PR, pmax) =
{
  my(Q = rkmap(a0, a, P, PTS));
  foreach(Q, z, print("PT\t", z[1], "\t", z[2]));
  print("PR\t", strjoin(apply(z -> Str(z), rksel(a, r, Q, PR, pmax)), ","));
}
