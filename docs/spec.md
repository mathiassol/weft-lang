# Weft — Language & Compiler Specification (v1)

**Status:** pre-implementation. This document is the plan; no compiler code exists yet.
**Owner:** Mathias Solheim
**Goal:** a resume-grade, finishable compiler project that reads as credible to compiler and hardware teams at ARM, Intel, and NVIDIA.

---

## 0. The one-sentence pitch

> **Weft** is a statically typed, data-oriented systems language with first-class SIMD vector types, compiled ahead-of-time to native x86-64 and AArch64 code through LLVM, with a one-command cross-platform installer.

The proof that it is real: a tiled single-precision matrix-multiply microkernel written **in Weft** that lands within 2x of the same algorithm written in C and compiled with `clang -O3`, on both AVX2 and NEON, with the generated assembly published in the README.

---

## 1. Name

### Candidates

| Name | Why it works | Why it might not |
| --- | --- | --- |
| **Weft** | Weaving term: the weft is the set of threads drawn *across* the warp. Maps exactly onto the language's core idea (lanes of data processed across an array). Four letters, one syllable, unclaimed in the programming-language space, so `weft-lang` becomes the canonical search result immediately. CLI `weft`, files `.weft`. | Slightly oblique — needs one sentence of explanation. That sentence is a *good* interview opener, not a cost. |
| **Stride** | Best pure semantic fit: stride is the memory-access concept at the heart of vectorization, so an Intel or ARM engineer understands the name before reading the description. | Collides with Stride3D, a reasonably well-known C# game engine, and with "stride" as generic English. You would never own the search results, and "Stride" alone on a resume is ambiguous. |
| **Lattice** | Strong double meaning: lattices are the algebraic structure underlying dataflow analysis, and lattices are how crystals (silicon) are described. Sounds technical. | Lattice Semiconductor is a real FPGA vendor. Putting "Lattice" on a resume sent to hardware companies invites a recruiter to misread it as employment or affiliation. Avoid. |

### Pick: **Weft**

Repository: `weft-lang`. Binary: `weft`. Source extension: `.weft`. Manifest: `weft.toml`.

**Why it reads well on a resume.** The bullet becomes:

> *Weft — designed and implemented a statically typed systems language with first-class SIMD vector types; hand-written lexer/parser/type checker in C++20, LLVM backend targeting x86-64 and AArch64, cross-platform toolchain with one-command install (Linux/macOS/Windows).*

The name is short enough not to eat the line, distinctive enough to be remembered between the resume screen and the interview, and unclaimed enough that "weft-lang" in a browser lands on Mathias's repo. It also survives the two questions every interviewer asks about a named project — "what does it mean?" and "why that name?" — with an answer that is about the language's design rather than about aesthetics.

---

## 2. Implementation language

### Pick: **C++20**, against the LLVM C++ API, built with CMake, tested with `lit` + `FileCheck`.

### Hiring-committee rationale

The people who will read this project are compiler engineers at three companies whose compiler work is, concretely:

- **ARM** — the Arm Compiler for Embedded and Arm's LLVM downstream; Arm is one of the largest upstream LLVM contributors (AArch64 backend, NEON/SVE/SME).
- **Intel** — ICX/oneAPI DPC++, all LLVM-derived; Intel owns huge parts of the X86 backend and the vectorizers.
- **NVIDIA** — NVVM/NVCC and the NVPTX backend, plus the CUDA front end.

All three are C++ LLVM codebases. Their intern interviews probe C++ and LLVM concepts directly: SSA form, `IRBuilder`, pass pipelines, `TargetMachine`, calling conventions, vector legalization. Writing Weft in C++ against LLVM's own headers means the project experience transfers *literally* — the same classes, the same CMake package (`find_package(LLVM CONFIG)`), and the same test harness (`lit`/`FileCheck`) those teams use every day. That is the difference between "I wrote a compiler" and "I work the way your team works."

A second, underrated signal: adopting `lit` + `FileCheck` for IR tests. Almost no student project does this, and every LLVM engineer recognises it instantly as a sign that someone has actually read the LLVM tree.

### The Rust case, stated fairly

Rust is the better *engineering* choice in isolation. `cargo` removes the cross-platform build problem, `inkwell` gives safe LLVM bindings, and `cargo-dist` would hand us the multi-OS installer and release matrix for free — most of Phase 4 below. ARM and NVIDIA both have growing Rust footprints. If the goal were "ship the most software with the least risk," the answer would be Rust.

It loses here for one reason: in Rust, the LLVM-facing code is thin glue over a C API, so the LLVM-internals signal is weaker exactly where it matters most, and the C++ signal — a hard requirement in essentially every compiler job description at these three companies — is absent. We are optimising for the hiring committee, and we pay for it with build engineering that Phase 0 exists to de-risk.

### What we do to neutralise Rust's advantages

- **CMake presets** (`CMakePresets.json`) so every platform builds with one command, no per-developer incantations.
- **LLVM acquisition is pinned and scripted, never "install LLVM somehow":** `apt.llvm.org` on Linux, `brew install llvm@<N>` on macOS, the official LLVM Windows release plus MSVC on Windows. One pinned major version, recorded in-repo and used identically by CI and by contributors.
- **Static linking for releases** so end users never install LLVM — they download one binary.
- **Hand-rolled installer** (`install.sh` + `install.ps1`, ~200 lines total). A bounded, one-time cost, and writing it yourself is itself a release-engineering talking point.

### The one tripwire that changes the plan

If a green Windows build that links LLVM's libraries cannot be achieved in CI during Phase 0, we do **not** switch languages. We move the backend boundary: the C++ compiler emits textual LLVM IR and drives a vendored `llc` + `lld` shipped in the release archive. Same codebase, same front end, slightly less LLVM-API surface, install story unchanged. Deciding this in advance is what stops Phase 0 from becoming a swamp.

---

## 3. Language scope for v1

Design brief: **small enough that one person finishes it, complete enough that nobody calls it a toy.** The yardstick is C plus modules plus SIMD, minus the unsafety papercuts — not Rust, not C++.

### 3.1 Syntax

Braces, `fn`, `let`/`var`, types after a colon, `->` for returns. Familiar on sight to anyone who knows Rust, Zig, Swift, or TypeScript, and cheap to parse with a hand-written recursive-descent parser (Pratt parsing for expressions).

```weft
module gemm.kernel

import core.simd
import core.io

const MR: i32 = 6;
const NR: i32 = 16;

struct Panel {
    data: []f32,
    rows: i32,
    cols: i32,
}

fn dot(a: []f32, b: []f32) -> f32 {
    var acc: f32x8 = f32x8.splat(0.0);
    var i: i32 = 0;
    while i + 8 <= a.len {
        acc = simd.fma(simd.load8(a, i), simd.load8(b, i), acc);
        i = i + 8;
    }
    var tail: f32 = simd.reduce_add(acc);
    while i < a.len {
        tail = tail + a[i] * b[i];
        i = i + 1;
    }
    return tail;
}

fn main() -> i32 {
    let arena = alloc.Arena.init(1 << 20);
    defer arena.deinit();

    let xs = arena.alloc_slice(f32, 1024);
    for i in 0..xs.len { xs[i] = f32(i); }

    io.print_f32(dot(xs, xs));
    return 0;
}
```

### 3.2 Type system

- **Scalars:** `i8 i16 i32 i64`, `u8 u16 u32 u64`, `f32 f64`, `bool`, `void`. Explicit widths only — no `int`.
- **Aggregates:** `struct` (C layout, `#[packed]` attribute available), fixed arrays `[N]T`, **slices** `[]T` (pointer + length, bounds-checked in debug builds, `-O` strips the checks), raw pointers `*T` and `*mut T`.
- **Enums:** C-style integer enums with an explicit backing type.
- **SIMD vectors:** `vec<N, T>` is the canonical form, with aliases `f32x4 f32x8 f64x2 f64x4 i32x4 i32x8 u8x16 …`. First-class values: elementwise arithmetic and comparison operators, `splat`, `shuffle`, `select`, `reduce_add/min/max`, aligned and unaligned load/store, `fma`. These lower straight to LLVM `<N x T>` vectors, which is what makes the cross-ISA story below honest.
- **Generics:** monomorphised generic functions and structs, one or more type parameters, **no trait bounds** — instantiate then type-check, C++-template-style but with far simpler rules. Enough for `Arena.alloc_slice(T, n)`, `Vec<T>`, and the SIMD helpers; nowhere near the cost of a real trait system.
- **Inference:** local only. `let x = expr;` infers from the initialiser. Function signatures, struct fields, and globals are always annotated. No Hindley–Milner, no bidirectional inference through generics.
- **Methods:** `impl Type { fn … }` with non-virtual, statically dispatched methods and an implicit `self`. No vtables, no interfaces.
- **Constants:** `const` with a compile-time constant-folding evaluator over integer/float/bool expressions. Not a full comptime interpreter.

### 3.3 Memory and errors

- **Manual memory, explicit allocators.** No GC, no reference counting, no ownership or borrow analysis. Allocators are passed as ordinary values (Zig's convention); the stdlib ships a page allocator and a bump/arena allocator.
- **`defer`** for scope-exit cleanup. Cheap to implement, removes most of the pain of manual cleanup, and reads as a deliberate design decision.
- **Errors:** explicit return values plus `panic()` / `assert()`, which trap with a message and a source location. No exceptions, no unwinding, no `Result` type in v1 (a `Result` needs tagged unions, which are a non-goal — see §4).
- **C ABI compatibility:** `extern "C"` declarations and exports, matching the platform calling convention. Weft can call libc and be called from C. This is what makes a small stdlib feasible and is a strong systems-programming signal on its own.

### 3.4 Modules

- One file = one module; `module a.b.c;` at the top must match the path under the source root.
- `import a.b.c;` or `import a.b.c as c;`. No glob imports, no cyclic imports (detected and reported).
- Visibility: private by default, `pub` to export. Two levels, no module-relative visibility rules.

### 3.5 LLVM code generation

- Typed AST → name resolution → type checking → **direct `IRBuilder` emission**, alloca-per-local, and let LLVM's `mem2reg`/SROA build SSA. This is the Kaleidoscope approach scaled up; it is honest, standard, and avoids hand-writing dominance frontiers for v1. A separate typed MIR is a v2 refactor, not a v1 requirement.
- Optimisation via `PassBuilder` with the stock `O0`/`O1`/`O2`/`O3` pipelines, selectable per build profile.
- Object emission through `TargetMachine`; `--emit=ll|asm|obj|exe` on every invocation so codegen is always inspectable. This flag is not a debugging convenience, it is a *portfolio feature* — it is how a reviewer verifies the claims.
- **Debug info:** `DIBuilder` line tables plus function and parameter scopes, DWARF on Linux/macOS and CodeView on Windows. Enough to set a breakpoint and step through a `.weft` file in `lldb`, `gdb`, or Visual Studio. Full type-level debug info is a non-goal.
- **Linking:** the compiler emits objects and then drives a linker. `lld` is vendored into the release archives for Linux and Windows; macOS uses the Xcode Command Line Tools linker because linking against the macOS SDK requires it. `weft doctor` checks for and explains every per-platform prerequisite instead of failing with a raw linker error.
- **SIMD targeting:** `--cpu=baseline|native|<name>` sets the LLVM target features. Baseline is SSE2 on x86-64 and NEON on AArch64; `x86-64-v3` unlocks AVX2/FMA. Because Weft's `vec<N, T>` maps to LLVM vector types rather than to intrinsics, LLVM's legaliser splits or widens vectors per target — so `f32x8` compiles to one `vfmadd231ps` on AVX2 and to two `fmla` instructions on NEON, from identical source.

### 3.6 Standard library kernel

Deliberately a *kernel*, not a batteries-included stdlib. Written in Weft on top of a thin `extern "C"` shim.

| Module | Contents |
| --- | --- |
| `core` | `assert`, `panic`, `mem.copy/set/compare`, slice helpers, numeric limits |
| `core.alloc` | page allocator (`mmap` / `VirtualAlloc`), `Arena`, `Vec<T>` |
| `core.simd` | aligned/unaligned load & store, `splat`, `shuffle`, `select`, `fma`, `reduce_*`, `min/max` |
| `core.math` | `sqrt`, `abs`, `floor`, `min`, `max`, `fma` — mapped to LLVM intrinsics |
| `core.io` | stdout/stderr writers, whole-file read/write |
| `core.fmt` | integer and float formatting into a caller-provided buffer |
| `core.time` | monotonic clock, for the benchmark harness |

No strings library beyond UTF-8 byte slices. No collections beyond `Vec<T>`. No threading.

---

## 4. Explicit non-goals for v1

Writing these down is what separates a scoped project from an abandoned one. Each is a legitimate feature that we are deliberately not building, and the README says so.

- **Self-hosting.** The Weft compiler will not be written in Weft.
- **Garbage collection, reference counting, ownership/borrow checking.** Manual memory, full stop.
- **Closures and lambdas.** Function pointers only.
- **Traits, interfaces, dynamic dispatch, operator overloading.**
- **Tagged unions / sum types and pattern matching**, and therefore no `Option`/`Result`.
- **Exceptions or unwinding.** `panic` traps.
- **`async`/`await`, threads, or any concurrency primitives.**
- **Macros or compile-time code execution** beyond constant folding.
- **Incremental or parallel compilation.** Whole-program, single-threaded front end.
- **A package registry.** Path dependencies and git dependencies pinned to a commit; no `npm` for Weft.
- **A GPU / NVPTX backend.** Explicitly v2 — and explicitly the single best follow-up for an NVIDIA application (see §9).
- **JIT or a REPL.** Ahead-of-time only.
- **Targets beyond x86-64 and AArch64.** No RISC-V, no WASM, no 32-bit, no cross-compilation in v1.
- **A borrow-free safety story.** Debug-mode bounds checks and traps are the extent of v1's safety claims, and the README says exactly that rather than implying more.

---

## 5. Phased path

Every phase ends in a state that is demoable and pushable. No phase depends on a later phase to be worth showing.

### Phase 0 — Decisions and skeleton (de-risk the build)

Repo, license, README, `CMakePresets.json`, pinned LLVM version, `lit` harness wired up, and the **full five-target CI matrix green on a trivial binary that links LLVM and prints its version**.

*Exit criteria:* `weft --version` builds and runs on linux-x86_64, linux-aarch64, macos-arm64, macos-x86_64, and windows-x86_64-msvc. If Windows is not green here, take the tripwire in §2 now rather than in Phase 3.

### Phase 1 — Foundation (front end + working codegen)

Lexer, Pratt/recursive-descent parser, AST, module graph and name resolution, type checker with real diagnostics, LLVM emission for scalars, structs, fixed arrays, slices, pointers, functions, all control flow, `defer`, C ABI interop, object emission and linking. `weft build`, `weft run`, `weft test`.

*Exit criteria:* non-trivial single-file programs (recursive fib, sieve of Eratosthenes, struct-heavy linked structures, a `qsort` over a slice) compile and produce identical output on all five targets; ≥60 `FileCheck` tests asserting the shape of the emitted IR; ≥80 golden-output run tests driven by `lit`.

### Phase 2 — SIMD, generics, and the stdlib kernel

`vec<N, T>` and the aliases, the `core.simd` operation set, monomorphised generics, methods, the allocator story, and the seven stdlib modules in §3.6. `--cpu` and `--emit` flags.

*Exit criteria:* the stdlib test suite passes on all five targets; `weft build --emit=asm --cpu=x86-64-v3` on a dot-product shows `vfmadd231ps`, and the same source on AArch64 shows `fmla`. Both snippets go in the README.

### Phase 3 — The demo, written in Weft

The hero program (§7) and one visual program, both written entirely in Weft, both in-repo under `examples/`, both run in CI.

*Exit criteria:* the GEMM kernel is within 2x of an equivalent C implementation at `clang -O3` on both AVX2 and NEON, and at least 10x faster than a naive triple loop; correctness is verified against a reference within tolerance; the benchmark table is generated by a script, not typed by hand.

### Phase 4 — Install and tooling

`install.sh` and `install.ps1` pulling checksummed release archives from GitHub Releases; `weft new`, `weft fmt`, `weft doctor`, `weft self update`; `weft.toml` manifest with path and pinned-git dependencies; `weft lsp` (diagnostics, hover types, document symbols, go-to-definition) and a VS Code extension with a TextMate grammar; a docs site on GitHub Pages.

*Exit criteria:* on a clean CI runner with no compiler and no LLVM installed, one command installs Weft and `weft run examples/mandelbrot` works — on all three operating systems. That test runs on every commit to `main`.

### Phase 5 — Polish and proof

Language reference, tutorial, an error-message quality pass (rustc-style spans, carets, and `note:`/`help:` lines), `CONTRIBUTING.md`, a tagged `v0.1.0` release with attached per-platform artifacts, a benchmark page, a 3-minute recorded demo, and the resume bullets.

---

## 6. Cross-platform, without lying

The claim is not "it works everywhere." The claim is a **published support-tier table that CI proves on every commit**, which is both stronger and honest.

### Tier 1 — tested on every commit to `main`

| Target | Runner | What runs |
| --- | --- | --- |
| `x86_64-linux-gnu` | `ubuntu-24.04` | build, `lit` suite, stdlib tests, demos, install-script test |
| `aarch64-linux-gnu` | `ubuntu-24.04-arm` | same |
| `aarch64-macos` | `macos-14` | same |
| `x86_64-macos` | `macos-13` (see note) | same |
| `x86_64-windows-msvc` | `windows-2022` | same |

All five are GitHub-hosted runners available to public repositories, so there is no self-hosted hardware dependency. `aarch64-windows-msvc` on `windows-11-arm` is a Tier-1 *stretch*; until it is green it is listed as Tier 2.

**Note on runner labels.** GitHub retires macOS images on a rolling basis, and the Intel macOS label in particular has moved (`macos-13` → `macos-15-intel`). Confirm the current Intel-macOS label at Phase 0 rather than trusting this table, and prefer the newest non-deprecated label. If GitHub ever drops hosted Intel macOS entirely, `x86_64-macos` moves to Tier 2 with a note — it does not get quietly claimed. The *targets* are the commitment; the runner labels are an implementation detail that CI owns.

### Tier 2 — builds, not continuously tested

Nothing in v1. The table stays empty until something earns a place in it. An empty Tier 2 is a better signal than an optimistic one.

### How the claim stays true

- **LLVM targets:** only `X86` and `AArch64` are registered. The compiler refuses an unknown `--target` with a clear message instead of producing broken output.
- **One-source-of-truth CI:** the same job definition runs on all five runners. A platform is not "supported" because it once worked on a laptop.
- **Install verified, not assumed.** The install script is exercised on a clean runner in the matrix, so "one-command install" is a tested assertion.
- **The demos are part of the matrix.** Cross-platform means the *SIMD* code works cross-platform, which is the interesting claim and the one most projects quietly skip.
- **Documented, honest prerequisites.** macOS needs the Xcode Command Line Tools for linking; Windows needs the MSVC build tools unless the vendored `lld-link` path is used. `weft doctor` detects and explains both. Saying this plainly is worth more than a false zero-dependency claim.

---

## 7. The demo program

### Hero: `examples/gemm` — a tiled SGEMM microkernel in Weft

A blocked single-precision matrix multiply with a register-blocked microkernel, written entirely in Weft:

- packed panel layout (the classic Goto-style repacking of A and B into contiguous panels)
- a register-blocked microkernel — 6x16 accumulating in `f32x8` on AVX2, 4x16 in `f32x4` on NEON
- cache blocking parameters exposed as `const`s so the tuning is visible and explainable
- a benchmark harness in Weft using `core.time`, reporting GFLOP/s
- correctness checked against a naive reference within a float tolerance
- `--emit=asm` output for the microkernel committed under `examples/gemm/asm/`, so a reviewer can see `vfmadd231ps` and `fmla` without building anything

**Why this demo and not something else.** GEMM is the single most-studied kernel in the industry these three companies occupy. Anyone on a compiler, libraries, or performance team at ARM, Intel, or NVIDIA has personally cared about a GEMM microkernel. Choosing it says: I know what your team actually does. It is also small — a few hundred lines — and it is *falsifiable*: GFLOP/s on two ISAs and a delta against `clang -O3` are numbers a skeptic can reproduce, which is the opposite of vaporware.

The resume bullet this produces:

> *Wrote a tiled, register-blocked SGEMM microkernel in Weft itself, reaching within 2x of `clang -O3` C on AVX2 and NEON — validating the language's SIMD types end-to-end through LLVM codegen.*

### Visual: `examples/mandelbrot`

An 8-lane SIMD Mandelbrot renderer writing a PPM, converted to PNG in CI and embedded at the top of the README. GEMM proves the engineering; this gives the repo a picture, which is what makes a stranger scroll down. It also exercises vector comparisons, `select`, and masked accumulation — a genuinely different codegen path from GEMM's FMA chain.

### Also shipped: `examples/` smoke set

Sieve, `qsort` over a slice, a word-frequency counter over a byte slice, an arena-allocated linked structure. Small, but they are what a reader skims to decide whether the language is pleasant.

---

## 8. Tooling and install

### CLI (`weft`)

| Command | Purpose |
| --- | --- |
| `weft new <name>` | scaffold a project with `weft.toml` and `src/main.weft` |
| `weft build [-O0..-O3] [--cpu=…] [--emit=ll\|asm\|obj\|exe]` | compile |
| `weft run` | build and execute |
| `weft test` | run `#[test]` functions in the current package |
| `weft fmt` | deterministic AST-based formatter, no configuration |
| `weft doctor` | check and explain per-platform prerequisites |
| `weft lsp` | LSP-lite server over stdio |
| `weft self update` | replace the installed toolchain in place |

### Modules and packages

`weft.toml` declares the package name, the Weft edition, and dependencies. Dependencies are **path** (`{ path = "../foo" }`) or **git pinned to a commit** (`{ git = "…", rev = "…" }`), vendored into `.weft/deps` and lockfiled in `weft.lock`. No registry, no version solver, no network calls during `weft build` once the lockfile is satisfied. This is the smallest package story that is still reproducible, and reproducibility is the part a reviewer cares about.

### Editor support

`weft lsp` is deliberately "LSP-lite": `publishDiagnostics`, `hover` (showing inferred types), `documentSymbol`, and `definition`. No completion, no rename, no refactorings. Paired with a ~100-line VS Code extension and a TextMate grammar, this produces squiggles and hover types in a real editor — an enormous jump in perceived quality for a small, bounded amount of work. A screen recording of live type errors in VS Code is worth more in a portfolio than any amount of prose.

### One-command install

```
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/mathiassol/weft-lang/main/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/mathiassol/weft-lang/main/install.ps1 | iex
```

Each script detects OS and architecture, downloads the matching archive from GitHub Releases, verifies its SHA-256 against a published manifest, unpacks to `~/.weft` (or `%LOCALAPPDATA%\weft`), and prints the exact `PATH` line to add. The archive contains the statically linked `weft` binary, the stdlib sources, and the vendored `lld` — no LLVM, no CMake, no compiler required on the user's machine. Homebrew tap and Scoop manifest are Phase 5 nice-to-haves, not the primary path.

**No "clone and hope."** The README's first code block is an install command, not `git clone`. Building from source is documented on a separate page, for contributors.

---

## 9. Why this is impressive to compiler and hardware intern hiring

Not because it is a language — plenty of students write toy languages — but because of *which* claims it makes verifiable:

- **It maps onto their actual job descriptions.** LLVM C++ API, `PassBuilder`, `TargetMachine`, calling conventions, vector legalisation, `lit`/`FileCheck`, DWARF and CodeView, a cross-OS release matrix. These are line items in ARM/Intel/NVIDIA compiler internship postings, and each one is a thing done rather than read about.
- **It produces performance numbers on two ISAs.** Almost every student compiler project stops at "it prints 42 on my laptop." Reaching within 2x of `clang -O3` on a GEMM microkernel, on both AVX2 and NEON, is a claim that requires the whole pipeline to genuinely work — and it is the kind of claim these particular companies find legible.
- **The demo is written in the language.** A compiler whose only test inputs are 20-line snippets is a parser. A compiler that builds a few hundred lines of performance-sensitive SIMD code, correctly, on five targets, is a compiler.
- **Everything is checkable in under two minutes.** Green CI badges across five targets, downloadable release binaries, committed assembly dumps, a reproducible benchmark script. A reviewer with limited time can *verify* rather than trust — which is exactly how a project avoids reading as vaporware.
- **The non-goals list demonstrates judgment.** Scoping is the skill being screened for in an intern, more than raw output. A spec that says "no closures, no GC, no sum types, no registry, and here is why" reads as engineering maturity.
- **There is an obvious, credible v2, per company.** An **NVPTX backend** so Weft kernels compile to PTX and run on a GPU (NVIDIA). **SVE/SME support with scalable vectors** (ARM). **A real autovectoriser, or an ISPC-style implicit-lane execution model** (Intel). Naming these in the spec and *not* building them in v1 is what proves the project is a foundation rather than a dead end — and it gives a specific, informed answer to "where would you take this next?" tailored to whoever is across the table.

---

## 10. Success criteria

v1 is done when every one of these is true. They are written to be checkable by someone other than the author.

| # | Criterion | Measure |
| --- | --- | --- |
| 1 | **Foundation of a full language** | Everything in §3 implemented and documented in the language reference; ≥60 `FileCheck` IR tests and ≥150 golden-output tests green; zero known-wrong-codegen bugs open |
| 2 | **A small project written in the language** | `examples/gemm` and `examples/mandelbrot` build and run on all five Tier-1 targets, in CI, from a clean checkout |
| 3 | **The demo is actually fast** | GEMM within 2x of the same algorithm in C at `clang -O3` on AVX2 *and* NEON; ≥10x the naive triple loop; numbers produced by a committed script |
| 4 | **Good install** | One command per OS, no prerequisites beyond the documented platform linker, under 60 seconds, verified on clean CI runners on every commit to `main` |
| 5 | **Good tooling** | `build/run/test/fmt/doctor/new/self update` all work; `weft fmt` is idempotent and enforced in CI; `weft lsp` gives diagnostics and hover types in VS Code |
| 6 | **Cross-platform, honestly** | All five Tier-1 targets green on every commit; support-tier table in the README matches CI exactly; no claimed target lacks a CI job |
| 7 | **Resume-grade** | Tagged `v0.1.0` with per-platform release artifacts and checksums; README with the Mandelbrot image, the benchmark table, and committed AVX2/NEON assembly; a docs site; a 3-minute demo video; three resume bullets that are all literally true |

### The minimum bar, restated in one line

A language with real syntax, a real type system, modules, SIMD, LLVM codegen, and a small stdlib — used to write a genuinely fast SIMD program — installable with one command on Linux, macOS, and Windows, with CI proving it on five targets.

Nothing in this document requires a second contributor, a GPU, paid infrastructure, or a research result. That is the point.
