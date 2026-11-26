.PHONY: build build-strict test test-fast test-asan build-asan asan-tests lint typecheck trusted clean
JOBS ?= 3
PYTHON ?= python3
CFLAGS ?= $(OPT)
WERROR ?=
include rankbound/native.mk
# the tests' runs on one core each, unless a test gives more (RANKBOUND_CORES: the default of --cores)
TESTENV = PYTHONDONTWRITEBYTECODE=1 RANKBOUND_CORES=1
TEST = $(TESTENV) $(PYTHON) -m unittest

# the prover's F_2 elimination, its kernel vectors (f2kernel) and siever; the checker's program, check rel and check rank (independent of the prover)
build: $(PROGRAMS)
$(PROGRAMS): %: %.c
	$(CC) $(CFLAGS) $(WARN) $(WERROR) $(THREADS) -o $@ $< $(LIBS)

# the same, rebuilt with warnings as errors (make test, CI); make build does not, as another compiler may warn
build-strict:
	$(MAKE) -B build WERROR="$(STRICT)"

# unit, reference (against PARI) and end-to-end tests, the modules in JOBS parallel shards (tests/shards.py); needs
# gp; about 4 min with JOBS=3 on a laptop (RANKBOUND_TEST_FULL=1: every case)
test: build-strict
	$(TESTENV) $(PYTHON) tests/shards.py --jobs $(JOBS)

# the same with fewer of the checker's random cases (RANKBOUND_TEST_FAST)
test-fast: build
	RANKBOUND_TEST_FAST=1 $(TESTENV) $(PYTHON) tests/shards.py --jobs $(JOBS)

# the C programs under AddressSanitizer and UBSan, through every test module that runs one (all but the pure-Python
# ones, ASAN_SKIP), then rebuilt as by make build.
SAN = -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
SANENV = ASAN_OPTIONS=detect_leaks=0:exitcode=99 UBSAN_OPTIONS=print_stacktrace=1 PYTHONPATH=tests:.
ASAN_SKIP = test_check_imports test_math_regressions test_math_speedups test_witness_stream test_check_form
ASAN_TESTS = $(filter-out $(ASAN_SKIP),$(basename $(notdir $(wildcard tests/test_*.py))))
test-asan:
	$(MAKE) -B build CFLAGS="$(SAN)"
	$(SANENV) $(TEST) $(ASAN_TESTS); s=$$?; rm -rf $(PROGRAMS:=.dSYM); $(MAKE) -B build; exit $$s

build-asan:
	$(MAKE) -B build CFLAGS="$(SAN)"

asan-tests:
	@echo $(ASAN_TESTS)

# lint: ruff with pyproject.toml's rules (pip install ruff)
lint:
	ruff check .

typecheck:
	mypy                # pyproject.toml's [tool.mypy] files

# the trusted base of a PASS (docs/verification.md): the lines of the checker's sources, which its hash covers
trusted:
	@cd rankbound/check && wc -l *.py *.gp *.c

clean:
	rm -rf $(PROGRAMS) build $$(find . -name __pycache__)
