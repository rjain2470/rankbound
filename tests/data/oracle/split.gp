\\ The relations of the 2-Selmer method, our SPLIT (rankbound/relations/split.py): the field K as the search sees it,
\\ the rows of the relations' odd valuations, and the relation search itself, the factor base relations (m1fb) and the
\\ triangular relations, one per degree-1 prime (m1tri, run in parallel by prime ranges).  Loaded after field.gp
\\ (sflabels); rankbound/selmer.gp builds on it.  Conventions as in field.gp: K = Q(al), al a root of f = F(x, 1), the
\\ GL2(Z)-reduced index form of O_K; a relation is one of
\\   [0, a, b]               the element a - b al (norm F(a, b)/lc(F))
\\   [1, c0, c1, c2, d]      the element (c0 + c1 al + c2 al^2)/d, in O_K
\\   [2, p]                  the ideal (p), the element p
\\ and the primes of K are labelled "p:r" (degree 1, r in 0..p, r = p for (1:0)) or "p:kK:fF" (k-th prime of
\\ idealprimedec, of degree F).  Output lines are tab-separated, numbers in decimal.

\\ ---------- the field

\\ K = [nf, al, f, the primes of lc(f), A (power basis of nf.pol -> coordinates on 1, al, al^2), J^-1, 0]
m1K(f, P) =
{
  my(nf, al, A);
  addprimes(P);
  [nf, al] = nfinit([f, concat([2, 3], P)], 3);
  A = matconcat(vector(3, i, Colrev(lift(al^(i - 1)), 3)))^-1;
  M1DEC = Map();
  [nf, al, f, factor(abs(pollead(f)))[, 1]~, A, idealhnf(nf, 1, al), 0];
}
\\ [idealprimedec(p), the labels of its primes], cached
m1dec(K, p) =
{
  my(d, dec, lab, sl);
  if(mapisdefined(M1DEC, p, &d), return(d));
  dec = idealprimedec(K[1], p); lab = vector(#dec, k, Str(p, ":k", k, ":f", dec[k][4]));
  sl = sflabels(K[1], K[2], K[6], K[3], p);
  foreach(sl, z, lab[z[2]] = Str(p, ":", z[1]));
  d = [dec, lab]; mapput(M1DEC, p, d); d;
}
\\ the element of K of a relation
m1el(K, r) = if(r[1] == 0, r[2] - r[3] * K[2], r[1] == 1, (r[2] + r[3] * K[2] + r[4] * K[2]^2) / r[5], r[2]);
\\ [c0, c1, c2, d] with el = (c0 + c1 al + c2 al^2)/d, d > 0
m1coords(K, el) =
{
  my(c = K[5] * Colrev(lift(Mod(el, K[1].pol)), 3), d = denominator(c));
  concat(Vec(c * d), [d]);
}
\\ the odd-valuation labels of the element el, whose valuations vanish outside the primes H and those of lc(f)
m1row(K, el, H) =
{
  my(cols = List());
  foreach(Set(concat(H, K[4])), p, my(d = m1dec(K, p), dec = d[1]);
    for(k = 1, #dec, if(idealval(K[1], el, dec[k]) % 2, listput(cols, d[2][k]))));
  Vec(cols);
}
\\ the primes of |n| with multiplicity
m1expand(n) = my(fa = factor(abs(n)), L = List()); for(i = 1, #fa~, for(t = 1, fa[i, 2], listput(L, fa[i, 1]))); Vec(L);
\\ one output line per relation: R kind data primes labels
m1print(tag, r, H, lab) =
{
  if(tag == "R", listput(M1REL, r));
  print(tag, "\t", strjoin(apply(z -> Str(z), r), ","), "\t", strjoin(apply(z -> Str(z), H), ","), "\t",
        strjoin(lab, ","));
}

\\ ---------- relations

\\ 1 if every prime of the factorisation fa is < B0 or in the set Ss
m1smooth(fa, B0, Ss) = for(i = 1, #fa~, my(q = fa[i, 1]); if(q >= B0 && !setsearch(Ss, q), return(0))); 1;
\\ the same, allowing one prime q in [B0, L) outside Ss to the first power (a large prime)
m1smoothlp(fa, B0, Ss, L) =
{
  my(nl = 0);
  for(i = 1, #fa~, my(q = fa[i, 1]);
    if(q >= B0 && !setsearch(Ss, q), if(nl || q >= L || fa[i, 2] > 1 || !isprime(q), return(0)); nl = 1));
  1;
}

\\ the element of the prime ideal pr (above p) of smallest search order whose norm is p^1 times a cofactor that
\\ passes ok(fa) (fa: the cofactor's factorisation); [element (zk coordinates), norm] or 0 after smax shells
m1split(K, pr, p, smax, ok) =
{
  my(nf = K[1], H = idealhnf(nf, pr), R, N, m, fa);
  R = H * qflll(nf[5][2] * H);
  for(s = 1, smax,
    forvec(c = vector(3, i, [-s, s]),
      if(vecmax(abs(c)) < s, next);
      my(t = 1); while(!c[t], t++); if(c[t] < 0, next);         \\ one of +-c
      my(el = R * c~);
      N = nfeltnorm(nf, el); m = abs(N) / p;
      if(m % p == 0, next);
      fa = factor(m, max(p, 3));
      if(ok(fa), return([el, N]))));
  0;
}

\\ for each degree-1 prime P above p in the set Ss with p >= pmin, nsplit elements of P (of smallest search order,
\\ at most smax shells in its LLL-reduced lattice) whose norm, factored by factor(, lim), passes ok(fa): prints "R ..."
\\ lines (kind 1), returns their count
m1sunits(K, Ss, pmin, lim, nsplit, smax, ok) =
{
  my(n = 0);
  foreach(Ss, p, if(p >= pmin, my(d = m1dec(K, p), dec = d[1]);
    for(k = 1, #dec, if(dec[k][4] == 1,
      my(H = idealhnf(K[1], dec[k]), R, got = 0);
      R = H * qflll(K[1][5][2] * H);
      for(s = 1, smax, if(got >= nsplit, break);
        forvec(cc = vector(3, i, [-s, s]), if(got >= nsplit, break);
          if(vecmax(abs(cc)) < s, next);
          my(t = 1); while(!cc[t], t++); if(cc[t] < 0, next);
          my(el = R * cc~, N = nfeltnorm(K[1], el), fa);
          if(N == 0, next);
          fa = factor(abs(N), lim);
          if(!ok(fa), next);
          my(r = concat([1], m1coords(K, nfbasistoalg(K[1], el))), H2 = m1expand(N));
          m1print("R", r, H2, m1row(K, m1el(K, r), H2)); n++; got++))))));
  n;
}

\\ the relations of the factor base: the free relations [2, p] for p < B0 and p in S; a - b al with F(a, b)
\\ smooth over FBP = {p < B0} u S but for at most one large prime below L, for max(|a|, b) in [slo, shi], until need of them (or a long run of shells without
\\ one); and for each degree-1 prime P above
\\ p in S with p >= B0, nsplit elements of P with FBP-smooth cofactor.  Prints "R ..." lines; returns their count.
m1fb(K, S, B0, need, slo, shi, nsplit, smax, L) =
{
  my(f = K[3], c = Vecrev(f), Ss = Set(S), n = 0, ok);
  forprime(p = 2, B0 - 1, if(!setsearch(Ss, p), m1print("R", [2, p], [p], m1row(K, p, [p])); n++));
  foreach(Ss, p, m1print("R", [2, p], [p], m1row(K, p, [p])); n++);
  n += m1sunits(K, Ss, B0, B0, nsplit, smax, fa -> m1smooth(fa, B0, Ss));
  my(cnt = 0, last = slo, top = slo - 1);
  for(s = slo, shi, if(cnt >= need || s > 2 * last + 40, break); top = s;
    for(b = 1, s, forstep(a = -s, s, if(b == s, 1, 2 * s), if(cnt >= need, break);
      if(gcd(a, b) != 1, next);
      my(N = c[1] * b^3 + c[2] * a * b^2 + c[3] * a^2 * b + c[4] * a^3, fa);
      if(N == 0, next);
      fa = factor(abs(N), B0);
      if(!m1smoothlp(fa, B0, Ss, L), next);
      my(H = m1expand(N));
      m1print("R", [0, a, b], H, m1row(K, a - b * K[2], H)); n++; cnt++; last = s)));
  print("FBN\t", cnt, "\t", top);
  n;
}

\\ the triangular relations for the degree-1 primes P above p in [lo, hi), p not in S nor dividing lc(f): one element of P with
\\ v_p(N) = 1 and every other prime of N below p.  Prints "T ..." lines, "NS lo hi count" (the degree-1 primes of
\\ the range, S included), "FREE p labels" (the row of (p) if p has a prime of degree > 1) and "FAIL p label" for a
\\ prime without a relation.
m1tri(f, P, S, lo, hi, smax) =
{
  my(K = m1K(f, P), Ss = Set(concat(S, K[4])), ns = 0);      \\ S and the primes of lc(f): no triangular relation
  forprime(p = lo, hi - 1, my(d = m1dec(K, p), dec = d[1]);
    if(#select(z -> z[4] > 1, dec), print("FREE\t", p, "\t", strjoin(m1row(K, p, [p]), ",")));
    for(k = 1, #dec, if(dec[k][4] != 1, next);
      ns++;
      if(setsearch(Ss, p), next);
      my(z = m1split(K, dec[k], p, smax, fa -> (#fa~ == 0 || vecmax(fa[, 1]) < p)));
      if(z == 0, print("FAIL\t", p, "\t", d[2][k]); next);
      my(r = concat([1], m1coords(K, nfbasistoalg(K[1], z[1]))), H = m1expand(z[2]));
      m1print("T", r, H, m1row(K, m1el(K, r), H))));
  print("NS\t", lo, "\t", hi, "\t", ns);
}
