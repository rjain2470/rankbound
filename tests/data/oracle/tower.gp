\\ The prover's half of the 2-isogeny descent (rankbound/tower.py), for curves E/Q with a rational point of order 2:
\\ the witnesses the witness stores (conic points and local points of the isogeny pairings, the Cassels-Tate data,
\\ the primes of the points) and tdprove, which prints everything the witness states.  The mathematics, the local
\\ images, the Selmer groups and the evaluation of the pairings are rankbound/verify/isogeny.gp's and ctp.gp's, loaded
\\ first (after verify/core.gp, for the characters of the points), so that the prover computes with the functions the
\\ verifier checks with; the Cassels-Tate data (tdct) and the local points (tdwit) are rankbound/ctp.gp's, loaded
\\ before this file.

\\ ---------- the isogeny pairings: the conic points

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

\\ ---------- the prover's local points (tdwit, tdsqwit: rankbound/ctp.gp)

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

\\ ---------- the isogeny pairings: the witnesses and the matrix

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

\\ ---------- the prover's Cassels-Tate data: rankbound/ctp.gp (tdct)

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

\\ the Cassels-Tate data tdct(mm, S) within left seconds (alarm), or 0 with a CTSKIP line (which: "E" or the isogeny)
\\ if it fails or runs out of time: the bound without it stands
tdctrun(which, mm, S, left) =
{
  if(left < 1, print("CTSKIP\t", which, "\te_ALARM: no time left"); return(0));
  my(r = iferr(alarm(left, tdct(mm, S)), e, e));         \\ alarm() returns its e_ALARM, other errors are caught
  if(type(r) == "t_ERROR", print("CTSKIP\t", which, "\t", errname(r), ": ", Str(r)); 0, r);
}

\\ the seconds left for the next pairing: ctsec (> 0) less the ctused seconds used, and at most ctend (>= 0) less
\\ the seconds since t0 (getabstime)
tdctleft(t0, ctsec, ctused, ctend) =
{
  my(x = ctsec - ctused);
  if(ctend >= 0, x = min(x, ctend - (getabstime() - t0) / 1000.));
  max(0, floor(x));
}

\\ everything the witness states, as lines for rankbound/tower.py: ai the model, P its bad primes, PTS
\\ points on it (or []), PR primes to try first for them, pmax the search limit, r0 the known lower bound on the rank:
\\ the Cassels-Tate pairings are computed (on E, then on each E') only while S > r0, within ctsec seconds in all
\\ (0: max(60 s, the time of the descent before them)) and ctend seconds from the start (< 0: no limit)
tdprove(ai, P, PTS, PR, pmax, r0, ctsec = 0, ctend = -1) =
{
  my(M, E, S = Set(concat([2], P)), w, D, Th, Bs, BS, CE = 0, CI, t0 = getabstime(), ctused = 0, t1);
  foreach(P, p, if(p < 2 || !ispseudoprime(p), error("The bad prime ", p, " is not prime.")));
  if(#Set(P) != #P, error("The bad prime ", select(p -> #select(q -> q == p, P) > 1, P)[1], " is listed twice."));
  M = tdmodel(ai, P); E = M[1];
  foreach(P, p, if(E.disc % p, error("The bad prime ", p, " does not divide the minimal discriminant.")));
  if(abs(E.disc) != prod(i = 1, #P, P[i]^valuation(E.disc, P[i])),
    error("The bad primes given do not account for the whole discriminant."));
  if(#M[3] == 0, error("tdprove: E(Q)[2] = 0"));
  w = ellrootno(E);
  D = vector(#M[3], i, tdisogeny(M[2], M[3][i], S));
  Th = vector(#D, i, my(d = D[i], ab2 = tdisog(d[2..3])); [tdpairing(d[2], d[3], S, d[4]), tdpairing(ab2[1], ab2[2], S, d[5])]);
  Bs = vector(#D, i, D[i][8] - Th[i][1][4] - Th[i][2][4]);
  CI = vector(#D);
  if(!ctsec, ctsec = max(60, ceil((getabstime() - t0) / 1000.)));      \\ the default: max(60 s, the descent's time)
  if(tdbound(Bs, w)[2] > r0,
    t1 = getabstime(); CE = tdctrun("E", E[1..5], S, tdctleft(t0, ctsec, ctused, ctend)); ctused += (getabstime() - t1) / 1000.;
    if(CE, Bs = vector(#D, i, min(Bs[i], D[i][8] - Th[i][1][4] - CE[5]))));
  for(i = 1, #D, if(tdbound(Bs, w)[2] <= r0, break);
    my(ab2 = tdisog(D[i][2..3]), E2 = ellminimalmodel(ellinit([0, ab2[1], 0, ab2[2], 0])));
    t1 = getabstime(); CI[i] = tdctrun(i, E2[1..5], S, tdctleft(t0, ctsec, ctused, ctend)); ctused += (getabstime() - t1) / 1000.;
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
