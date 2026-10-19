.PHONY: build test test-fast cado clean
JOBS ?= 3
CADO = rankbound/cado

# the prover's F_2 elimination and the verifier's dense rank (independent of each other)
build: rankbound/f2elim rankbound/verify/f2rank
rankbound/f2elim: rankbound/f2elim.c
	cc -O3 -pthread -o $@ $<
rankbound/verify/f2rank: rankbound/verify/f2rank.c
	cc -O3 -pthread -o $@ $<

# unit, reference (against PARI) and end-to-end tests; needs gp; ~3 min
test: build
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v

# the same without the las run of ICARM #1 (TestLas, ~1 min)
test-fast: build
	RANKBOUND_TEST_FAST=1 PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v

# CADO-NFS las and makefb into rankbound/cado/, with VERSION (tarball and sha256): make cado CADO_TARBALL=...
# Needs cmake >= 3.18, GMP (GMP=<prefix> if not on the default paths) and C++20 (macOS clang: CXXFLAGS="-std=c++20
# -fexperimental-library").  flask and requests get empty stand-ins for cmake.  CADO uses the build CPU's newest
# instructions, so build on the oldest CPU it will run on.
cado:
	@test -n "$(CADO_TARBALL)" || { echo 'usage: make cado CADO_TARBALL=cado-nfs.tar.gz'; exit 1; }
	rm -rf build/cado-src build/cado-build && mkdir -p build/cado-src build/cado-stub $(CADO)
	tar -xzf $(CADO_TARBALL) -C build/cado-src --strip-components 1
	for m in flask requests; do python3 -c "import $$m" 2>/dev/null || : > build/cado-stub/$$m.py; done
	PYTHONPATH=$(CURDIR)/build/cado-stub cmake -S build/cado-src -B build/cado-build -DCMAKE_BUILD_TYPE=Release > $(CADO)/cmake.log 2>&1
	$(MAKE) -C build/cado-build -j $(JOBS) las makefb > $(CADO)/make.log 2>&1
	cp build/cado-build/sieve/las build/cado-build/sieve/makefb $(CADO)/
	printf 'cado-nfs %s sha256 %s built %s\n' "$$(basename $(CADO_TARBALL))" \
	  "$$( (shasum -a 256 $(CADO_TARBALL) 2>/dev/null || sha256sum $(CADO_TARBALL)) | cut -d' ' -f1)" "$$(date +%F)" | tee $(CADO)/VERSION

clean:
	rm -rf rankbound/f2elim rankbound/verify/f2rank build $$(find . -name __pycache__)
