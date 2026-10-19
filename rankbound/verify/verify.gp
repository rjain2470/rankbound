\\ The number theory of rankbound.verify, also loaded by the prover's field.gp: the GRH bound (bdf), the relation
\\ checker (ck) and the field-level checks (v).

\\ Belabas, Diaz y Diaz, Friedman, "Small generators of the ideal class group", Math. Comp. 77 (2008), Corollary 2.2,
\\ inequality (10): if for some T > 1
\\   2 sum_{p, m: Np^m < T} log Np / Np^(m/2) * (1 - log Np^m / log T)
\\       > log |d_K| - n (gamma + log 8 pi - c1 / log T) - r1 (pi/2 - c2 / log T),   c1 = pi^2/2, c2 = 4 Catalan,
\\ then the prime ideals of norm < T generate Cl(K), assuming GRH for the L-functions of the nontrivial characters
\\ of Cl(K).  The proof works character by character, so for Cl(K)/2 the quadratic characters suffice: the zeta
\\ functions of the unramified quadratic extensions of K.  PARI's GRHchk (buch2.c) makes the same test.  It need not
\\ be monotone in T; the verifier checks it at the T used.

\\ splitting data [p, log p, [f_1, ..]] for the primes p < C
bdfdata(nf, C) =
{
  my(L = List()); forprime(p = 2, C - 1, listput(L, [p, log(p), apply(q -> q.f, idealprimedec(nf, p))])); Vec(L);
}

\\ 1 if (10) holds at T, in floating point with a margin of 1e-8 (D: bdfdata up to at least T)
bdfok(nf, D, T) =
{
  my(N = poldegree(nf.pol), R1 = nf.sign[1], logT = log(T), SA = 0., SB = 0.,
     cD = log(abs(nf.disc)) - N * (Euler + log(8 * Pi)) - R1 * Pi / 2, cN = N * Pi^2 / 2 + R1 * 4 * Catalan);
  if(#D && D[#D][1] < T - 1 && nextprime(D[#D][1] + 1) < T, error("bdfok: splitting data stops below T"));
  for(i = 1, #D, my(p = D[i][1], lp = D[i][2]); if(p >= T, break);
    foreach(D[i][3], f, my(logNP = f * lp); if(logNP >= logT, next);
      my(q = 1 / sqrt(p^f), M = ceil(logT / logNP) - 1, A = logNP * q, B = logNP * A);
      \\ sum over m = 1..M (Np^m < T) of logNP q^m and of logNP * m logNP q^m, in closed form
      if(M > 1, my(i1 = 1 / (1 - q)); A *= (1 - q^M) * i1; B *= (1 - q^M * (M + 1 - M * q)) * i1^2);
      SA += A; SB += B));
  cD + (cN + 2 * SB) / logT - 2 * SA < -1e-8;
}

\\ the least integer T with bdfok(T), by doubling and then bisection; an error if T would exceed 2 Tmax
bdf(nf, Tmax) =
{
  my(lo = 1, hi = 16, D);
  D = bdfdata(nf, hi);
  while(!bdfok(nf, D, hi), lo = hi; hi *= 2; if(hi > 2 * Tmax, error("BDF bound > Tmax")); D = bdfdata(nf, hi));
  while(hi - lo > 1, my(t = (lo + hi) \ 2); if(bdfok(nf, D, t), hi = t, lo = t));
  hi;
}

\\ The verifier's check of (10) in exact interval arithmetic: each real number becomes rationals lo <= x <= hi with
\\ denominator 2^BDFK.  Logarithms come from the atanh series with its tail bounded, Np^(-m/2) from sqrtint, and pi,
\\ Euler's and Catalan's constants and log pi from 40 decimal digits.
BDFK = 128;
\\ pi, Euler, Catalan, log pi, each truncated to 40 digits: c in [C, C + 10^-40]
bdfconst() =
{
  [31415926535897932384626433832795028841971, 5772156649015328606065120900824024310421,
   9159655941772190150546035149323841107741, 11447298858494001741434273513530587116472] / 10^40;
}

\\ [lo, hi] with lo <= 2 atanh z <= hi, 0 <= z <= 1/3 rational: the series 2 sum z^(2j+1)/(2j+1), whose tail after
\\ the last term taken is below 2 t / (1 - z^2) <= 9/4 t (t = the first term left out, without its 1/(2j+1))
bdfat(z) =
{
  my(s = 0, t = z, j = 0, S = 2^BDFK);
  while(t > 2^(-BDFK - 4), s += t / (2 * j + 1); t *= z^2; j++);
  [floor(2 * s * S), ceil((2 * s + 9 / 4 * t) * S)] / S;
}

\\ [lo, hi] with lo <= log n <= hi, n >= 1 an integer (l2 = bdfat(1/3), the bounds for log 2):
\\ log n = m log 2 + 2 atanh((n - 2^m) / (n + 2^m)), 2^m <= n < 2^(m+1)
bdflog(n, l2) = my(m = logint(n, 2)); m * l2 + bdfat((n - 2^m) / (n + 2^m));

\\ 1 if (10) holds at T, proved: with SA, SB as in bdfok, it reads (all terms positive)
\\   log T log |d_K| + cN + 2 SB < log T (n (gamma + 3 log 2 + log pi) + r1 pi / 2) + 2 SA log T,
\\ checked with upper bounds on the left and lower bounds on the right.
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

\\ The relation checker: recomputes the odd valuations of every relation in PARI.  K = Q(alpha) with alpha a root of
\\ f = F(x, 1); P lists the primes of disc(f).  A relation [a, b, H] is the element a - b alpha, with H the primes
\\ of F(a, b) with multiplicity; it needs prod H = |F(a, b)| != 0 and every h in H prime.  Since N(a - b alpha) =
\\ F(a, b)/lc(f), its valuation can be nonzero only above H or above lc(f).  Above p not dividing lc(f) the
\\ valuations are >= 0 with sum f_P v_P = v_p(F(a, b)), so the loop stops once that sum is reached.  A free relation
\\ [p] is the ideal (p).  Output: "R i" and the odd-valuation primes as p:k (k: index in idealprimedec) or p:k:f.

ckinit(f, P) =
{
  my(nf, al); addprimes(P); [nf, al] = nfinit([f, concat([2, 3], P)], 3);
  if(#nfcertify(nf), error("ckinit: maximal order not certified"));
  if(subst(f, x, al) != 0, error("ckinit: alpha is not a root of f"));
  [nf, al, factor(abs(pollead(f)))[, 1]~];
}
ckdec(D, nf, p) = my(d); if(mapisdefined(D, p, &d), d, d = idealprimedec(nf, p); mapput(D, p, d); d);
ckcol(p, k, P) = if(P.f == 1, Str(p, ":", k), Str(p, ":", k, ":", P.f));

\\ rels: vector of [a, b, H] or [p]; i0: index of rels[1]; returns the number checked, errors on a failed check
ckrels(K, f, rels, i0) =
{
  my(nf = K[1], al = K[2], Jl = K[3], D = Map(), F2 = substvec(f, [x], [x / y]) * y^3, lc = pollead(f));
  for(i = 1, #rels, my(r = rels[i], cols = List());
    if(#r == 1,
      my(p = r[1], dec); if(!isprime(p), error("free relation: ", p, " not prime"));
      dec = ckdec(D, nf, p); for(k = 1, #dec, if(dec[k].e % 2, listput(cols, ckcol(p, k, dec[k])))),
      my(a = r[1], b = r[2], H = r[3], n = subst(subst(F2, x, a), y, b), el = a - b * al, ps);
      if(n == 0, error("relation ", i0 + i - 1, ": F(a,b) = 0"));
      if(abs(n) != vecprod(H), error("relation ", i0 + i - 1, ": prod H != |F(a,b)|"));
      foreach(Set(H), h, if(!isprime(h), error("relation ", i0 + i - 1, ": ", h, " not prime")));
      ps = Set(concat(H, Jl));
      foreach(ps, p, my(dec = ckdec(D, nf, p), full = (lc % p != 0), vp = if(full, valuation(n, p), 0), s = 0);
        for(k = 1, #dec, my(v = idealval(nf, el, dec[k]));
          if(v % 2, listput(cols, ckcol(p, k, dec[k])));
          if(full, s += v * dec[k].f; if(s == vp, break)))));
    print("R ", i0 + i - 1, " ", strjoin(Vec(cols), " ")));
  #rels;
}

\\ the small columns: number of degree-1 primes of norm < GB
ckns(K, GB) = my(ns = 0); forprime(p = 2, GB - 1, foreach(idealprimedec(K[1], p), P, ns += (P.f == 1))); ns;

\\ Brumer-Kramer terms [u, n] (KSW19 Prop. 3.1): E minimal, nf its 2-division field, P the primes of disc(E)
vbk(E, nf, P) =
{
  my(D = E.disc, n = 0);
  foreach(P, p, my(red = elllocalred(E, p));
    if(red[1] == 1, n += (valuation(D, p) % 2 == 0), red[1] >= 2, n += #idealprimedec(nf, p) - 1));
  [if(D > 0, 2, 1), n];
}

\\ The field-level checks: each claim is recomputed and printed as "CHK name 0|1 info".  ai: the input model; P: the
\\ claimed bad primes; f = F(x, 1): the claimed index form; then the claimed d_K, GB, ns, u, n, w.

vchk(name, ok, info = "") = print("CHK ", name, " ", if(ok, 1, 0), " ", info);

vfield(ai, P, mm, f, dKc, GBc, nsc, uc, nc, wc) =
{
  my(E, D, T, nf, K, ok, u, n, w, ns);
  foreach(P, p, if(!isprime(p), error("verify: ", p, " is not prime")));
  vchk("bad_primes_prime", 1, Str(#P, " primes, isprime (APRCL)"));
  addprimes(P); E = ellminimalmodel(ellinit(ai)); D = E.disc;
  vchk("minimal_model", E[1..5] == mm);
  vchk("disc_factored", abs(D) == prod(i = 1, #P, P[i]^valuation(D, P[i])) && #Set(P) == #P
       && #select(p -> D % p, P) == 0, "the bad primes are exactly the primes of the minimal discriminant");
  T = x^3 + E.b2*x^2 + 8*E.b4*x + 16*E.b6;
  vchk("E2_trivial", polisirreducible(T), "2-division cubic irreducible: E(Q)[2] = 0");
  nf = nfinit([T, concat([2, 3], P)]);
  vchk("maximal_order", #nfcertify(nf) == 0 && nf.disc == dKc, "d_K");
  vchk("index_form", poldegree(f) == 3 && poldisc(f) == dKc && content(f) == 1, "disc F = d_K, so R_F = O_K");
  K = ckinit(f, P);
  vchk("same_field", nfisisom(K[1], nf) != 0);
  [u, n] = vbk(E, nf, P);
  vchk("bk_u", u == uc, Str("u = ", u));
  vchk("bk_n", n == nc, Str("n = ", n));
  w = ellrootno(E); vchk("root_number", w == wc, Str("w = ", w));
  \\ the GRH generation bound: BDF Corollary 2.2 holds at GB (on K = Q(alpha), isomorphic to nf)
  vchk("bdf_GB", GBc > 1 && bdfcert(K[1], bdfdata(K[1], GBc), GBc),
       Str("BDF (10) holds at T = ", GBc, " (exact interval arithmetic)"));
  ns = ckns(K, GBc); vchk("ns", ns == nsc, Str("degree-1 primes of norm < GB: ", ns));
  K;
}

\\ rank lower bound: r points, F2-independent by their 2-signatures at primes p with #E(F_p) = 2 mod 4
vrank(ai, r, PTS, PR) =
{
  my(E0 = ellinit(ai), ok, rows = []);
  ok = #PTS == r && #Set(PR) == #PR;
  for(i = 1, #PTS, ok = ok && ellisoncurve(E0, PTS[i]));
  for(k = 1, #PR, my(p = PR[k], Ep, n, h, row); ok = ok && isprime(p) && p >= 3 && E0.disc % p; if(!ok, break);
    Ep = ellinit(E0[1..5], p); n = ellcard(Ep); ok = ok && n % 4 == 2; h = n / 2;
    row = vector(#PTS, i, if(denominator(PTS[i][1]) % p == 0 || denominator(PTS[i][2]) % p == 0, 0,
      my(Q = [Mod(PTS[i][1], p), Mod(PTS[i][2], p)]); if(ellmul(Ep, Q, h) == [0], 0, 1)));
    rows = concat(rows, [row]));
  \\ r = 0 needs no point (rank >= 0)
  vchk("rank_lower_bound", ok && (r == 0 || (#rows && matrank(Mod(matconcat(rows~), 2)) == r)), Str(r, " points, 2-signatures at ", #PR, " primes"));
}
