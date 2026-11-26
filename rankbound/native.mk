# The one list of rankbound's native programs and their compiler flags: read by the Makefile (include) and by setup.py
# (which parses these simple NAME = value lines; keep them that way).  Each program P is built from P.c.
PROGRAMS = rankbound/f2elim rankbound/f2kernel rankbound/check/check rankbound/sieve
OPT = -O3
WARN = -Wall -Wextra
STRICT = -Werror
THREADS = -pthread
LIBS = -lm
