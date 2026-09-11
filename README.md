# Weft

**A statically typed, data-oriented systems language with first-class SIMD vector types, compiled ahead-of-time to native x86-64 and AArch64 code through LLVM.**

> ### Status: pre-implementation (spec-first)
>
> **There is no compiler here yet.** This repository currently contains the design specification and roadmap, published *before* implementation on purpose. Nothing in this README describes working software unless it says so.
>
> No benchmark numbers, no CI badges, and no support claims will appear below until CI can prove them. When a number shows up here, it will be produced by a committed script on a hosted runner.

Weft is a personal flagship project by [Mathias Solheim](https://github.com/mathiassol), aimed squarely at compiler and toolchain work: LLVM code generation, SIMD/vector types that lower cleanly across ISAs, and a cross-platform toolchain that a stranger can install with one command.

---

## Read the spec

**[`docs/spec.md`](docs/spec.md)** — the full v1 specification: name and stack rationale, the type system, the module system, LLVM codegen strategy, the stdlib kernel, explicit non-goals, the phased plan, the cross-platform support-tier policy, tooling and install design, and the success criteria.

## What Weft is meant to be

- **Small on purpose.** Roughly "C, plus modules, plus SIMD" — not a Rust or C++ competitor. Explicit widths (`i32`, `f32`), `struct`, fixed arrays, slices, pointers, manual memory with explicit allocators, and `defer`.
- **SIMD in the type system, not in an intrinsics header.** `vec<N, T>` with aliases like `f32x8` and `i32x4`, lowered to LLVM vector types so identical source compiles to AVX2 on x86-64 and NEON on AArch64 via LLVM's legaliser.
- **Codegen you can see.** `--emit=ll|asm|obj|exe` is a shipped feature, not a debug flag: it is how a reader verifies the claims rather than trusting them.
- **Cross-platform as a hard requirement.** Linux, macOS, and Windows, on both ISAs where hosted runners exist, with a published support-tier table that must match what CI actually runs.
- **One-command install.** A shell or PowerShell one-liner that fetches a checksummed, statically linked release binary. No `git clone` as the primary path, and no requirement that the user install LLVM.

## Planned syntax

Illustrative, from the spec — this does not compile yet.

```weft
module gemm.kernel

import core.simd
import core.io

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
```

## Roadmap

| Phase | Work | Exit criteria |
| --- | --- | --- |
| **0** | Repo, CMake presets, pinned LLVM, `lit` harness, full CI matrix | `weft --version` builds and runs on all five Tier-1 targets |
| **1** | Lexer, parser, modules, type checker, LLVM codegen, linking, `build`/`run`/`test` | Non-trivial programs produce identical output on every target; IR asserted with `FileCheck` |
| **2** | `vec<N, T>`, `core.simd`, monomorphised generics, methods, stdlib kernel | Stdlib green everywhere; `--emit=asm` shows `vfmadd231ps` on AVX2 and `fmla` on NEON |
| **3** | The demo, written **in Weft**: a tiled register-blocked SGEMM microkernel, plus a SIMD Mandelbrot renderer | GEMM within 2x of the same algorithm in C at `clang -O3` on both ISAs, and ≥10x a naive triple loop |
| **4** | `install.sh` / `install.ps1`, releases with checksums, `new`/`fmt`/`doctor`/`self update`, `weft.toml`, LSP-lite + VS Code extension | One command installs Weft and runs an example on a clean runner, on all three operating systems |
| **5** | Language reference, tutorial, diagnostics quality pass, tagged `v0.1.0` with artifacts | Every success criterion in the spec is checkable by someone else |

## Non-goals for v1

Stated up front, because scope is the whole game for a solo project: no self-hosting, no GC or borrow checking, no closures, no traits or dynamic dispatch, no sum types or pattern matching (and therefore no `Option`/`Result`), no exceptions, no concurrency, no macros or compile-time execution, no incremental compilation, no package registry, no GPU/NVPTX backend, no JIT or REPL, and no targets beyond x86-64 and AArch64.

The reasoning for each is in [the spec](docs/spec.md). An NVPTX backend, ARM SVE/SME scalable vectors, and a typed MIR with its own dataflow passes are the intended directions *after* v1.

## Planned platform support

Tier 1 means tested in CI on every commit to `main`. None of these are green yet — this is the target, and the table will not claim a target that lacks a CI job.

| Target | Runner | State |
| --- | --- | --- |
| `x86_64-linux-gnu` | `ubuntu-24.04` | planned |
| `aarch64-linux-gnu` | `ubuntu-24.04-arm` | planned |
| `aarch64-macos` | `macos-14` | planned |
| `x86_64-macos` | `macos-13` | planned |
| `x86_64-windows-msvc` | `windows-2022` | planned |

## Implementation stack

C++20 against the LLVM C++ API, built with CMake, tested with LLVM's own `lit` + `FileCheck` harness. The rationale — including an honest account of why Rust would have been the lower-risk engineering choice, and the pre-agreed fallback if linking LLVM on Windows proves painful — is in [§2 of the spec](docs/spec.md).

## License

[MIT](LICENSE)
