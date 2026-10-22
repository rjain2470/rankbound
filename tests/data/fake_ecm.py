"""A stand-in for GMP-ECM in tests (test_factor.py): `fake_ecm.py -q -one -maxmem M (-c K B1 | -pm1 B1)`, the number
on stdin.  It finds the prime p of FAKE_ECM_FIND="p:B1,..." (or "p:B1-B1max") when B1 >= that B1 (and <= B1max, and p
divides the number), else runs
K curves of FAKE_ECM_SPC seconds each and finds nothing; FAKE_ECM_FAIL=1 exits 1.  Each call appends its arguments to
FAKE_ECM_LOG."""
import os, sys, time

a = sys.argv[1:]
n = int(sys.stdin.read().split()[0])
with open(os.environ['FAKE_ECM_LOG'], 'a') as f:
    f.write(' '.join(a) + ' ' + str(len(str(n))) + '\n')
if os.environ.get('FAKE_ECM_FAIL'):
    print('fake error'); sys.exit(1)
pm1 = '-pm1' in a
b1 = float(a[a.index('-pm1') + 1] if pm1 else a[a.index('-c') + 2])
k = 1 if pm1 else int(a[a.index('-c') + 1])
for x in os.environ.get('FAKE_ECM_FIND', '').split(','):
    if x:
        p, at = x.split(':')
        lo, hi = (at.split('-') + [at])[:2] if '-' in at else (at, 'inf')
        if n % int(p) == 0 and int(p) != n and float(lo) <= b1 <= float(hi):
            time.sleep(float(os.environ.get('FAKE_ECM_SPC', '0')))
            print(p, n // int(p))
            sys.exit(6)                     # GMP-ECM: a probable prime factor, composite cofactor
time.sleep(k * float(os.environ.get('FAKE_ECM_SPC', '0')))
print(n)
sys.exit(0)
