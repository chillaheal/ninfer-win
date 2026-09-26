# Fix flash_next real-model load (token_embedding [248320,2560] vs [256,256])

## Context
`ninfer-serve` (and CLI) fail to load the real 75 GB flash_next model:
`[bind-diag] text/token_embedding MISMATCH got shape=[248320,2560] want shape=[256,256]` →
`ArtifactError: tensor descriptor does not match target contract: text/token_embedding`.

Root cause: `Package::construct_loaded_model` (impl/package.cpp:42) calls the **mini**
`LoadedModel::open` (4-arg, [256,256] contract). The real 75 GB model is [248320,2560].
The real loader (`real_loader.cpp`) and real Program factory (`create_real_program`,
program.cpp:1569) exist but are **dead code** — the `Package` never invokes them.

## Fix — wire the real path into the Package
1. **real_loader.h/.cpp** — store `artifact_path_`/`ngram_path_` in `open`; add
   `artifact_path()` / `ngram_path()` accessors.
2. **package.h** — `using LoadedModel = qwen3_8_flash_next::RealLoadedModel;`
   (forward decl `class RealLoadedModel;`); change `create_program` to take
   `std::unique_ptr<LoadedModel>` (ownership) so the real H2D model is moved into
   the Program exactly once (no double H2D → no OOM).
3. **package.cpp** — include `impl/load/real_loader.h`;
   `construct_loaded_model` → `RealLoadedModel::open(artifact_path, ngram_path, device)`;
   `create_program` → `create_real_program_from_model(std::move(model), plan.capacity(),
   device, ops::QsaIndexerKvDtype::Fp8)`.
4. **program.cpp / family.h** — add `create_real_program_from_loaded(
   std::unique_ptr<RealLoadedModel>, max_context, device, idx_dtype)` (no re-open);
   existing `create_real_program(artifact, ngram, ...)` delegates to it.
5. **registry.cpp:497** — `create_program(std::move(loaded->model), ...)`.

## Verify
- Rebuild: `cmd //c "tools\flash_next_dev\p1_rebuild_tests.bat"` then
  `cmd //c "tools\flash_next_dev\p1_build.bat"` (visible terminal).
- `ctest -L flash_next` (mini tests must stay green — they use the mini path directly).
- Real test (detached, visible terminal, never in-session): stop 27b → start
  flash_next on the new build → stop it → start 27b → fix anything broken.
