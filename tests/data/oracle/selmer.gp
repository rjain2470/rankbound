\\ rankbound 3, M1: the data of the exact 2-Selmer bound for E/Q with E(Q)[2] = 0 (rankbound/selmer.py does the F_2
\\ algebra and most classes and characters, exactly as these functions define them): the characters and local classes
\\ of a relation (the definitions, and what Python leaves to GP: m1batch), the local images and the setup.  Loaded after
\\ field.gp and relations/split.gp (the field, the relations and their search), with its conventions.  Output lines are
\\ tab-separated, numbers in decimal.

\\ ---------- characters and local classes of the relations

\\ the Legendre characters at Q = [q, r] (F_2: 1 iff a non-residue) of the relation r
m1chi(r, Q) =
{
  my(q = Q[1], z = Q[2], v);
  v = if(r[1] == 0, r[2] - r[3] * z, r[1] == 1, (r[2] + r[3] * z + r[4] * z^2) * Mod(r[5], q)^-1, r[2]);
  v = Mod(v, q);
  if(v == 0, error("m1chi: ", q, " divides a relation"));
  kronecker(lift(v), q) < 0;
}
\\ the signs (F_2: 1 iff negative) of the relation r at the real roots RR of f
m1sgn(r, RR) =
{
  vector(#RR, i, my(z = RR[i], v);
    v = if(r[1] == 0, r[2] - r[3] * z, r[1] == 1, r[2] + r[3] * z + r[4] * z^2, r[2]);
    if(abs(v) < 10^-30, error("m1sgn: a relation is too close to 0 at a real place"));
    v < 0);
}
\\ the places of S: [p, pr, kind data] for every prime pr above p in S; kind data: [nfmodprinit, q = N(pr)] for odd
\\ p, the bid of pr^(2e+1) for p = 2
m1places(K, S) =
{
  my(L = List());
  foreach(S, p, my(dec = m1dec(K, p)[1]);
    foreach(dec, pr, listput(L, [p, pr, if(p == 2, idealstar(K[1], idealpow(K[1], pr, 2 * pr[3] + 1)),
                                            [nfmodprinit(K[1], pr), p^pr[4]])])));
  Vec(L);
}
\\ the class of el != 0 in K_P*/K_P*^2 at one place: [v mod 2, the unit's class bits]
m1cls1(K, el, pl) =
{
  my(y, v = nfeltval(K[1], el, pl[2], &y), out = [v % 2]);
  if(pl[1] == 2, my(lg = ideallog(K[1], y, pl[3]), cy = pl[3].cyc);
    for(i = 1, #cy, if(cy[i] % 2 == 0, out = concat(out, [lg[i] % 2]))),
    my(z = nfmodpr(K[1], y, pl[3][1]));
    out = concat(out, [if(z^((pl[3][2] - 1) / 2) == 1, 0, 1)]));
  out;
}
\\ the classes at every place of S, concatenated
m1cls(K, el, PL) = concat(vector(#PL, i, m1cls1(K, el, PL[i])));

\\ ---------- the local images

\\ 1 if the rational t != 0 is a square in Q_p (exact: t is given exactly)
m1issq(t, p) =
{
  my(v, u);
  if(t == 0, return(0));
  v = valuation(t, p); if(v % 2, return(0));
  u = t / p^v;
  if(p == 2, (numerator(u) * denominator(u)) % 8 == 1, kronecker(numerator(u) * denominator(u), p) == 1);
}
\\ the F_2 rank of the rows of the vector of vectors L
m1rank(L) = if(#L, matrank(Mod(matconcat(L~), 2)), 0);
\\ centers for m1points: rational approximations (mod p^m) of the roots in Q_p of W and of W'.  Points on the non-
\\ identity components of the Neron model lie p-adically close to the singular point of the reduction, where W and W'
\\ have their roots (at depth about v_p(disc W)/2 for I_n), which uniform samples almost never reach.
m1centers(W, p, m) =
{
  my(C = List());
  foreach([W, deriv(W, 'x)], g, foreach(polrootspadic(g, p, m), r, if(valuation(r, p) >= 0, listput(C, lift(truncate(r))))));
  Vec(C);
}
\\ points x in Q with W(x) a nonzero square in Q_p whose classes of x - th at the places PLp above p span a space of
\\ dimension t; [the points, their classes].  Every other x is near a center (m1centers) at a random depth j <= J.
m1points(K, W, th, p, PLp, t, tries = 200000) =
{
  my(X = List(), C = List(), rk = 0, k = 4, J = valuation(poldisc(W), p) \ 2 + 4, Z);
  if(t == 0, return([[], []]));
  Z = m1centers(W, p, 2 * J + 10);
  for(i = 1, tries,
    if(i % 200 == 0, k++);
    my(xx, w, cl);
    xx = if(#Z && i % 2, Z[random(#Z) + 1] + p^(random(J) + 1) * (random(p^4) + 1) * (-1)^random(2),
            random(p^k) / p^(2 * random(2)) * (-1)^random(2));
    w = subst(W, 'x, xx);
    if(!m1issq(w, p), next);
    cl = concat(vector(#PLp, l, m1cls1(K, xx - th, PLp[l])));
    if(m1rank(concat(Vec(C), [cl])) > rk, listput(X, xx); listput(C, cl); rk++);
    if(rk == t, return([Vec(X), Vec(C)])));
  error("m1points: the local image at ", p, " has rank ", rk, " < ", t, " after ", tries, " tries");
}
\\ the point at infinity's image: one x with W(x) > 0 between the two smaller real roots, if W has three; its
\\ signs at the real roots RR of f
m1inf(W, tc, RR) =
{
  my(e, xx);
  if(#RR < 3, return([[], []]));
  e = vecsort(real(polroots(W)));
  xx = bestappr((e[1] + e[2]) / 2, 2^20);
  if(!(xx > e[1] && xx < e[2]) || subst(W, 'x, xx) <= 0, error("m1inf: no point between the real roots"));
  [[xx], [vector(#RR, i, xx - (tc[1] + tc[2] * RR[i] + tc[3] * RR[i]^2) < 0)]];
}

\\ ---------- the data of the Python classes (selmer.py's Places): per place, what makes the class of a - b al or of an
\\ integer exact without GP.  "PLD i p e f t g cp ct tab": i the place's index in m1places order (0-based); t empty if
\\ p does not divide lc(f), else an integer with al' = 1/(t - al) pr-integral, and then a - b al = (a' - b' al') (t - al)
\\ with a' = b, b' = t b - a; g the minimal polynomial over F_p of the image of al (or al') in the residue field
\\ (coefficients low to high, monic, of degree dividing f); cp the class of p, ct that of t - al (empty without t);
\\ tab, for p = 2 only, the classes of a0 - b0 al (al' if t), a0, b0 in 0..7, entry 8 b0 + a0, "-" where it is not a
\\ unit at pr.  Above 2 the class of a pr-unit depends only on its class mod 8 O_pr (8 O_pr in pr^(2e+1)).  rtab: below.
m1pld(K, pl, i) =
{
  my(nf = K[1], al = K[2], f = K[3], p = pl[1], pr = pl[2], t = "", be = al, g, ct = "", tab = "", mp = nfmodprinit(nf, pr));
  if(pollead(f) % p == 0,
    t = 0; while(nfeltval(nf, t - al, pr) > 0, t++);
    be = 1 / (t - al); ct = strjoin(apply(z -> Str(z), m1cls1(K, t - al, pl)), ""));
  g = minpoly(nfmodpr(nf, be, mp));
  if(type(g) != "t_POL", g = 'x - g);
  g = Vecrev(apply(z -> lift(Mod(z, p)), g));
  if(p == 2,
    my(L = List());
    for(b0 = 0, 7, for(a0 = 0, 7, my(el = a0 - b0 * be);
      listput(L, if(el != 0 && nfeltval(nf, el, pr) == 0, strjoin(apply(z -> Str(z), m1cls1(K, el, pl)), ""), "-"))));
    tab = strjoin(Vec(L), ","));
  \\ rtab, for a ramified pr above a small p: the classes of a0 - b0 al (al'), a0, b0 mod p^M, entry p^M b0 + a0, where
  \\ v = v_pr(a0 - b0 al) has v + c <= e M (c = 2e + 1 above 2, else 1): then every a - b al congruent to it mod p^M
  \\ has that valuation and a unit part congruent mod pr^c, so the same class
  my(rtab = "", M = if(pr[3] < 2, 0, p == 2, 4, p <= 64, 1, 0));
  if(M, my(L = List(), q = p^M, c = if(p == 2, 2 * pr[3] + 1, 1));
    for(b0 = 0, q - 1, for(a0 = 0, q - 1, my(el = a0 - b0 * be, v);
      listput(L, if(el == 0, "-", v = nfeltval(nf, el, pr); if(v + c <= pr[3] * M,
                    strjoin(apply(z -> Str(z), m1cls1(K, el, pl)), ""), "-")))));
    rtab = Str(M, ":", strjoin(Vec(L), ",")));
  print("PLD\t", i - 1, "\t", p, "\t", pr[3], "\t", pr[4], "\t", t, "\t", strjoin(apply(z -> Str(z), g), ","), "\t",
        strjoin(apply(z -> Str(z), m1cls1(K, p, pl)), ""), "\t", ct, "\t", tab, "\t", rtab);
}
\\ the real roots of f in increasing order, each as N with the root in [N, N + 1] / 2^M1RQ
M1RQ = 640;
m1realq(f) = { my(R); localbitprec(1200); R = polrootsreal(f); apply(z -> floor(z * 2^M1RQ), R); }
\\ the character primes: nq degree-1 Q = (q, al - r), q >= qmin (so q divides no norm), q not in S nor lc(f)
m1chars(f, S, qmin, nq) =
{
  my(Q = List(), q = max(qmin, 3));
  while(#Q < nq, q = nextprime(q + 1);
    if(setsearch(S, q) || pollead(f) % q == 0, next);
    my(rt = polrootsmod(f, q)); if(#rt, listput(Q, [q, lift(rt[1])])));
  Vec(Q);
}

\\ ---------- the setup

\\ Everything of the curve with minimal model ai, index form f and bad primes P but the relations: theta, the real
\\ places, the places of S (the primes Sx: 2 and the bad primes, or selmer.py's REDUCE_S choice) with their data, the character primes (nq, from qmin), the local images.  Returns K.
\\ theta, W, the signature, the real roots and the places of S (the primes Sx) with #S_K: what every format-4 witness
\\ needs; returns [K, W, theta, its coordinates, the real roots, S]
m1head(ai, f, P, Sx) =
{
  my(K, E, W, th, S, RR, nsk = 0);
  K = m1K(f, P);
  E = ellinit(ai); W = 'x^3 + E.b2 * 'x^2 + 8 * E.b4 * 'x + 16 * E.b6;
  \\ theta: a root of W in K, on 1, al, al^2
  th = nfisisom(W, K[1]);
  if(#th == 0, error("m1head: W has no root in K"));
  th = Mod(th[1], K[1].pol);
  if(subst(W, 'x, th) != 0, error("m1head: theta"));
  my(tc = Vec(K[5] * Colrev(lift(th), 3)));
  print("TH\t", strjoin(apply(z -> Str(z), tc), ","));
  print("W\t", strjoin(apply(z -> Str(z), Vecrev(W)), ","));
  print("SIG\t", K[1].sign[1], "\t", K[1].sign[2]);
  \\ the real roots of f in increasing order, indexed from 0
  localprec(200);
  my(rts = polroots(f));
  RR = vecsort(real(select(z -> imag(z) == 0, rts)));
  print("REAL\t", strjoin(vector(#RR, i, Str(i - 1)), ","));
  S = Set(Sx);
  print("SPRIMES\t", strjoin(apply(z -> Str(z), S), ","));
  foreach(S, p, my(dec = m1dec(K, p)[1], t = #select(z -> z[3] == 1 && z[4] == 1, dec));
    nsk += #dec;
    print("PLACE\t", p, "\t", #dec, "\t", if(t == 3, 2, t) + (p == 2), "\t", strjoin(m1dec(K, p)[2], ",")));
  print("SK\t", nsk);
  [K, W, th, tc, RR, S];
}

\\ the local images: points of E(R) and E(Q_p) for p in S whose classes span them ("PT" lines)
m1pts(K, W, th, tc, RR, S, PL) =
{
  my(inf = m1inf(W, tc, RR));
  print("PT\tinf\t", strjoin(apply(z -> Str(z), inf[1]), ","), "\t",
        strjoin(apply(z -> strjoin(apply(b -> Str(b), z), ""), inf[2]), ","));
  foreach(S, p, my(PLp = select(z -> z[1] == p, PL), dec = m1dec(K, p)[1], t = #select(z -> z[3] == 1 && z[4] == 1, dec), pt);
    pt = m1points(K, W, th, p, PLp, if(t == 3, 2, t) + (p == 2));
    print("PT\t", p, "\t", strjoin(apply(z -> Str(z), pt[1]), ","), "\t",
          strjoin(apply(z -> strjoin(apply(b -> Str(b), z), ""), pt[2]), ",")));
}
m1base(ai, f, P, qmin, nq, Sx) =
{
  my(K, W, th, tc, RR, S, PL, Q);
  [K, W, th, tc, RR, S] = m1head(ai, f, P, Sx);
  print("REALQ\t", M1RQ, "\t", strjoin(apply(z -> Str(z), m1realq(f)), ","));
  Q = m1chars(f, S, qmin, nq);
  print("CHARS\t", strjoin(apply(z -> Str(z[1], ":", z[2]), Q), ","));
  PL = m1places(K, S);
  print("CLS\t", strjoin(apply(pl -> Str(pl[1], ":", #m1cls1(K, 1, pl)), PL), ","));
  for(i = 1, #PL, m1pld(K, PL[i], i));
  m1pts(K, W, th, tc, RR, S, PL);
  K;
}

\\ m1base with our SPLIT's factor base relations ("R" lines, relations/split.gp's m1fb).  B0: the factor base bound;
\\ need, slo, shi: the a,b relations wanted, and the shells max(|a|, b) searched; nq: the character primes, from
\\ max(B0, L); L: the large prime bound of the a,b relations (<= B0: none).
m1setup(ai, f, P, B0, need, slo, shi, nsplit, smax, nq, L, Sx) =
{
  my(K = m1base(ai, f, P, max(B0, L), nq, Sx), S = Set(Sx), ns0 = 0);
  \\ the factor base: the degree-1 primes below B0
  forprime(p = 2, B0 - 1, my(d = m1dec(K, p)); for(k = 1, #d[1], if(d[1][k][4] == 1, ns0++; print("FB\t", d[2][k]))));
  print("NS0\t", ns0);
  M1REL = List();
  \\ the factor base takes the primes of lc(f) as it takes S: the J part of every a - b al lies above them, so they
  \\ get no triangular relation (m1tri skips them too)
  m1fb(K, Set(concat(Vec(S), K[4])), B0, need, slo, shi, nsplit, smax, L);
}

\\ m1base for the sieve's relations (selmer.py's sieve_witness), with elements of the degree-1 primes above the p
\\ in S beyond the relations (p >= pmin), their norms smooth over the primes below lim and S ("R" lines, m1sunits)
m1sieve(ai, f, P, qmin, nq, pmin, lim, nsplit, smax, Sx) =
{
  my(K = m1base(ai, f, P, qmin, nq, Sx), S = Set(Sx));
  M1REL = List();
  m1sunits(K, S, pmin, lim, nsplit, smax, fa -> m1smooth(fa, lim, S));
}

\\ What selmer.py's classes and characters leave to GP, for the relations in the file rf (one vector per line, as
\\ m1el reads them): items [j, i, [place indices, 0-based]] print "CL j t bits" (the class of relation i at place t);
\\ items [j, i, -1] print "CH j bits" (its signs at the real roots, then its characters at Q).
m1batch(f, P, rf, items, Q, Sx) =
{
  my(K = m1K(f, P), S = Set(Sx), PL, R = readvec(rf), RR);
  PL = m1places(K, S);
  localprec(200); RR = vecsort(real(select(z -> imag(z) == 0, polroots(f))));
  foreach(items, it, my(r = R[it[2]], el = m1el(K, r));
    if(type(it[3]) == "t_INT",
      print("CH\t", it[1], "\t", strjoin(apply(z -> Str(z), concat(m1sgn(r, RR), apply(z -> m1chi(r, z), Q))), "")),
      foreach(it[3], t, print("CL\t", it[1], "\t", t, "\t", strjoin(apply(z -> Str(z), m1cls1(K, el, PL[t + 1])), "")))));
}
