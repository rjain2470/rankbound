\\ TEST ONLY (tests/test_selmer.py): build the data of a format-4 witness by other means than the verifier and the
\\ prover: the class group, S-units and Cl_S from bnfinit / bnfsunit, dim Sel_2 from ell2cover.  Loaded after
\\ rankbound/field.gp (for bdf only) and rankbound/verify/core.gp and selmer.gp (for vbk only).  Never loaded by the verifier.

mkhex(N) = my(F = factor(N), L = List()); for(i = 1, #F~, for(m = 1, F[i, 2], listput(L, Strprintf("%x", F[i, 1])))); strjoin(Vec(L), ",");
\\ the coordinates of the nf element z in 1, al, al^2
mkco(nf, Minv, z) = Minv * nfalgtobasis(nf, z);
\\ the "e:" line of z
mke(nf, Minv, z) =
{
  my(c = mkco(nf, Minv, z), d = denominator(c), N = nfeltnorm(nf, z));
  c = c * d; Str("e:", c[1], ",", c[2], ",", c[3], ",", d, ":", mkhex(numerator(abs(N))));
}
\\ field.gp's label [p, r] of the degree-1 prime pr (r - al, or 1 for r = p, times J), or 0 if none
mklabel(nf, al, f, p, pr) =
{
  my(Ji = idealhnf(nf, 1, al), rs = apply(z -> lift(z), Vec(polrootsmod(f, p))));
  if(pollead(f) % p == 0, rs = concat(rs, [p]));
  foreach(rs, r, my(el = if(r < p, r - al, 1)); if(idealval(nf, el, pr) - idealval(nf, Ji, pr) > 0, return(r)));
  -1;
}
\\ W(x) a nonzero square in Q_p (p = 0: W(x) > 0)
mksq(T, p, xx) =
{
  my(w = subst(T, x, xx), v, t);
  if(w == 0, return(0)); if(p == 0, return(w > 0));
  v = valuation(w, p); if(v % 2, return(0)); t = numerator(w / p^v) * denominator(w / p^v);
  if(p == 2, t % 8 == 1, kronecker(t, p) == 1);
}
\\ candidate local points at p: integers, small denominators, points near each p-adic root of T
mkpts(T, p, nmax) =
{
  my(L = List(), C = List(), rts = polrootspadic(T, p, 30));
  for(t = -60, 60, listput(C, t));
  for(t = -40, 40, if(t % p, listput(C, t / p^2); listput(C, t / p^4)));
  foreach(rts, e, my(e0 = truncate(e)); for(kk = 0, 6, for(t = 1, 12, listput(C, e0 + t * p^kk); listput(C, e0 - t * p^kk))));
  foreach(C, xx, if(#L < nmax && mksq(T, p, xx) && !setsearch(Set(L), xx), listput(L, xx)));
  Vec(L);
}

\\ print the data of the witness of E (ai) by lines "TAG value"; mink: the Minkowski bound (else BDF)
mk4(ai, mink) =
{
  my(E, P, SP, T, nfT, Bm, Bi, g, g2, co, Iform, f, nf, al, Minv, dK, r1, r2, B, ns, lc, th, Sp, cols, bnf, su, gens,
     R = List(), M, lg, sm, dimV, G, sco = List(), Mu, k, dimCl, dimCls, nSK, U, nonS, Mn, Ker, chr = List(), chq = List(),
     rcv, el = List(), ech, rk, u, n, w, Sbk, dsel, S);
  E = ellminimalmodel(ellinit(ai)); P = factor(abs(E.disc))[, 1]~; SP = Set(concat([2], P));
  T = x^3 + E.b2*x^2 + 8*E.b4*x + 16*E.b6;
  nfT = nfinit(T); dK = nfT.disc;
  \\ the index form (Delone-Faddeev), not reduced
  Bm = matrix(3, 3, i, kk, polcoef(nfT.zk[kk], i - 1)); Bi = Bm^-1;
  g = 'u * nfT.zk[2] + 'v * nfT.zk[3]; g2 = lift(Mod(g, nfT.pol)^2);
  co = (z -> Bi * vector(3, i, polcoef(z, i - 1, x))~);
  Iform = substvec(matdet(matconcat([co(1), co(g), co(g2)])), ['u, 'v], [x, 1]);
  f = if(pollead(Iform) < 0, -Iform, Iform);
  if(poldisc(f) != dK || poldegree(f) != 3, error("mk4: index form"));
  [nf, al] = nfinit([f, concat([2, 3], P)], 3);
  Minv = matconcat([nfalgtobasis(nf, 1), nfalgtobasis(nf, al), nfalgtobasis(nf, al^2)])^-1;
  if(nfT.pol != T, error("mk4: nfinit changed T"));
  th = Mod(nfisisom(nfT, nf)[1], nf.pol);
  if(subst(T, x, th) != 0, error("mk4: theta"));
  r1 = nf.sign[1]; r2 = nf.sign[2]; lc = pollead(f);
  B = if(mink, floor((4 / Pi)^r2 * 2 / 9 * sqrt(abs(dK))) + 1, bdf(nf, floor(12 * log(abs(dK))^2)));
  ns = 0; forprime(p = 2, B - 1, foreach(idealprimedec(nf, p), pr, ns += pr.f == 1));
  \\ the columns: every prime above S, below B and above lc
  Sp = Set(concat([SP, primes([2, B - 1]), factor(lc)[, 1]~]));
  cols = List(); foreach(Sp, p, foreach(idealprimedec(nf, p), pr, listput(cols, [p, pr]))); cols = Vec(cols);
  bnf = bnfinit(nf, 1);
  su = bnfsunit(bnf, apply(z -> z[2], cols));
  gens = concat([[bnf.tu[2]], bnf.fu, su[1]]);
  \\ the relations: [line, element]: the S'-units, free relations at S, and a few a - b al smooth over Sp
  foreach(gens, z, listput(R, [mke(nf, Minv, z), z]));
  foreach(SP, p, listput(R, [Str("f:", p), p]));
  my(F2 = substvec(f, [x], [x / 'y]) * 'y^3, nab = 0);
  for(b = 1, 6, for(a = -12, 12, my(Fab);
    if(nab >= 8 || gcd(a, b) != 1, next); Fab = subst(subst(F2, x, a), 'y, b);
    if(Fab == 0, next);
    my(m = abs(Fab)); foreach(Sp, p, m /= p^valuation(m, p));
    if(m == 1, nab++; listput(R, [Str(a, ",", b, ":", mkhex(abs(Fab))), a - b * al]))));
  R = Vec(R);
  M = matrix(#R, #cols, i, j, idealval(nf, R[i][2], cols[j][2]) % 2);
  lg = select(j -> !(cols[j][2].f == 1 && cols[j][1] < B), [1..#cols]);
  dimV = matrank(Mod(M, 2)) - matrank(Mod(vecextract(M, "..", lg), 2));
  G = ns - dimV;
  \\ s_columns: every degree-1 prime above S, with its label
  for(j = 1, #cols, my(p = cols[j][1], pr = cols[j][2]);
    if(setsearch(SP, p) && pr.f == 1, my(r = mklabel(nf, al, f, p, pr)); if(r < 0, error("mk4: label")); listput(sco, [p, r, j])));
  Mu = matconcat([M; matrix(#sco, #cols, i, j, j == sco[i][3])]);
  k = ns - (matrank(Mod(Mu, 2)) - matrank(Mod(vecextract(Mu, "..", lg), 2)));
  dimCl = #select(d -> d % 2 == 0, bnf.cyc);
  my(SK = List()); foreach(SP, p, foreach(idealprimedec(nf, p), pr, listput(SK, pr)));
  nSK = #SK; dimCls = #select(d -> d % 2 == 0, bnfsunit(bnf, Vec(SK))[5][2]);
  if(G != dimCl || k != dimCls, error("mk4: G = ", G, " k = ", k, " but Cl/2 = ", dimCl, " Cl_S/2 = ", dimCls));
  U = r1 + r2 + nSK + k;
  \\ the elements: combinations of relations with even valuations outside S, independent by characters
  nonS = select(j -> !setsearch(SP, cols[j][1]), [1..#cols]);
  Mn = vecextract(M, "..", nonS);
  Ker = if(#nonS, lift(matker(Mod(Mn~, 2))), matid(#R));
  my(rts, NR = apply(z -> nfeltnorm(nf, z[2]), R), bad = 1);
  localbitprec(600); rts = polrootsreal(f);
  foreach(NR, N, bad *= numerator(N) * denominator(N));
  foreach(R, z, bad *= denominator(mkco(nf, Minv, z[2])));
  for(i = 1, r1, listput(chr, i - 1));
  forprime(q = 3, 4000, if(#chq >= 3 * U + 10, break);
    if(setsearch(SP, q) || lc % q == 0 || bad % q == 0, next);
    my(rs = polrootsmod(f, q)); if(#rs, listput(chq, [q, lift(rs[1])])));
  \\ the character vector of each relation
  rcv = vector(#R, i, my(z = R[i][2], c = mkco(nf, Minv, z), gp = c[1] + c[2] * x + c[3] * x^2);
    concat(vector(r1, kk, real(subst(gp, x, rts[kk])) < 0),
           vector(#chq, kk, my(q = chq[kk][1], cc = chq[kk][2], d = denominator(c), t = (c[1] + c[2] * cc + c[3] * cc^2) * d^2);
             (1 - kronecker(numerator(t) * denominator(t) % q, q)) / 2)));
  \\ mix the kernel basis (column t plus column t + 1, invertible) so that elements combine several relations
  Ker = matconcat(vector(#Ker, t, if(t < #Ker, (Ker[, t] + Ker[, t + 1]) % 2, Ker[, t])));
  ech = matrix(0, r1 + #chq); rk = 0;
  for(t = 1, #Ker, my(cv = Ker[, t], ch = sum(i = 1, #R, cv[i] * rcv[i]) % 2, M2 = matconcat([ech; ch]));
    if(rk < U && matrank(Mod(M2, 2)) > rk, ech = M2; rk++; listput(el, apply(i -> i - 1, select(i -> cv[i], [1..#R])))));
  if(rk != U, error("mk4: v = ", rk, " < U = ", U));
  [u, n] = vbk(E, nfT, P); w = ellrootno(E);
  Sbk = G + u + n; if((-1)^Sbk != w, Sbk--);
  dsel = #ell2cover(ellinit(E[1..5]));
  S = min(dsel, Sbk);
  print("MM ", strjoin(apply(z -> Str(z), Vec(E[1..5])), ","));
  print("BAD ", strjoin(apply(z -> Str(z), P), ","));
  print("POLY ", strjoin(apply(z -> Str(z), Vecrev(Vec(f))), ","));
  print("DK ", dK); print("B ", B); print("NS ", ns);
  print("W ", strjoin(apply(z -> Str(z), Vecrev(Vec(T))), ","));
  print("TH ", strjoin(apply(z -> Str(z), Vec(mkco(nf, Minv, th))), ","));
  print("SPR ", strjoin(apply(z -> Str(z), SP), ","));
  print("TORS ", strjoin(apply(z -> Str(z), elltors(E)[2]), ","));
  foreach(R, z, print("REL ", z[1]));
  foreach(sco, z, print("SCOL ", z[1], " ", z[2]));
  foreach(el, J, print("EL ", strjoin(apply(z -> Str(z), J), ",")));
  print("CHR ", strjoin(apply(z -> Str(z), Vec(chr)), ","));
  foreach(chq, z, print("CHQ ", z[1], " ", z[2]));
  foreach(SP, p, print("LOC ", p, " ", strjoin(apply(z -> Str(z), mkpts(T, p, 24)), ",")));
  my(er = polrootsreal(T), XI = List());
  if(#er == 3, foreach([floor(er[3]) + 1, floor(er[3]) + 7, bestappr((er[1] + er[2]) / 2, 1000), bestappr((2 * er[1] + er[2]) / 3, 1000)],
    xx, if(mksq(T, 0, xx), listput(XI, xx))), listput(XI, floor(er[1]) + 2));
  print("INF ", strjoin(apply(z -> Str(z), Vec(XI)), ","));
  print("CL G ", G); print("CL dimV ", dimV); print("CL k ", k); print("CL U ", U); print("CL v ", rk);
  print("CL dim_sel_cap_V ", dsel); print("CL S_bk ", Sbk); print("CL S ", S); print("CL w ", w);
}
