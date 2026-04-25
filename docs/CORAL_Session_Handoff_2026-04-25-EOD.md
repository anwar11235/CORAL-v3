# CORAL Session Handoff — 2026-04-25

> **Update 2026-04-25 EOD:** This handoff was written at session start (before the 2026-04-25 work session). Significant developments during the session — warm-start regression and fix, eval Fix A regression and revert, dataset determinism discovery and fix — have updated the project state materially. See `docs/CORAL_Session_Handoff_2026-04-25_EOD.md` for the current state as of end of day.

**Purpose:** Context carry-over for the next Claude session. The previous session (2026-04-24) launched and completed the Phase 3c training run. This document captures the result, the mechanism findings, the strategic reframing of the experimental roadmap, and the open items.

**Status as of this handoff:** Phase 3c run complete. New CORAL best on Sudoku-Extreme-1K at 67.62%, single seed. Vast instance destroyed. All artifacts pulled to local Windows. Multi-seed campaign + within-Sudoku compounding curve experiment are the priorities for the next session.

---

## The Phase 3c result

**Run:** `phase3c_option_y`, W&B `voc0vjxs`, branch `moe-lb-specialization` HEAD `dbf21ea`. Launched 2026-04-24 23:20 UTC, completed 2026-04-25 04:35 UTC. ~5h15m runtime, ~$3.20 GPU cost on a Sweden A100 SXM4 40GB instance.

**Final eval at step 52080: 67.62% on Sudoku-Extreme-1K.** New CORAL best, 1.57pp above satisfied-owl (66.05%).

**Trajectory was non-monotonic** — Phase 3c was behind for most of the run before pulling ahead in the final ~10k steps:

| Step | Phase 3c | satisfied-owl | Gap |
|------|----------|---------------|-----|
| 5208 | 28.80% | ~34% | -5pp (consolidation shock) |
| 10416 | 41.07% | ~42% | -1pp |
| ~15624 | ~43% | ~46% | -3pp |
| 20832 | 51.26% | ~51% | 0pp (parity) |
| ~26040 | 56.52% | 56.2% | +0.3pp |
| 36456 | 61.83% | 64.01% | -2.2pp |
| **52080** | **67.62%** | 66.05% | **+1.57pp** |

**Single seed.** The 1.57pp gap is within plausible single-seed variance for a run of this length. Multi-seed is required before claiming Phase 3c > satisfied-owl as a real architectural win rather than a lucky draw. Until multi-seed lands, the honest framing is *"Phase 3c is at least competitive with satisfied-owl, possibly better."*

**Local artifacts:** `C:\Users\mauha\dev-projects\coral_v3_results\phase3c_option_y\`
- `phase3c_option_y_step52080.pt` (116 MB) — best checkpoint
- `phase3c_option_y.log` (3.4 MB) — full training log
- `run-20260424_232025-voc0vjxs/` — wandb local run dir with metrics history, config, requirements

---

## Mechanism findings

The interesting story isn't the headline number — it's that Option Y's L_lb formulation produced *genuine sparse expert specialization*, not the mode collapse or uniform-routing collapse I feared during the run.

**Per-step behavior (train metrics):**
- `routing_entropy` stable at ~0.1 throughout (vs max ln(33) ≈ 3.5) — router is decisive per step
- `codebook_utilisation_frac` 0.03-0.06 — 1-2 modes per step out of 32
- `mean_passthrough_weight` ~0.01 — codebook used hard, not bypassed
- `recon_loss` stable at ~1e-4 — codebook reconstructs cleanly

**Across-batch behavior:**
- `lb_loss` decayed from 0.0023 (post-consolidation peak) to 0.0006 (step 21k) — global mode usage became balanced over time
- This is consistent with different inputs being routed to different small expert subsets, with no single expert dominating across the dataset

**Joint read:** The router achieves per-input sparse specialization while maintaining balanced global utilization. This is the classical "successful sparse MoE" regime, not the failure modes (single-mode collapse, uniform averaging, passthrough dominance) that satisfied-owl and earlier configs landed in.

The Phase 3c codebook is doing real work — encoding 32 distinct spatial templates and routing inputs to the appropriate subset. Whether this contributes to the 1.57pp accuracy gain over satisfied-owl, or whether it's a coincidence with the headline result, requires multi-seed to disambiguate.

---

## Strategic reframing of the experimental roadmap

A late-session conversation reframed the commercial thesis and consequently the experimental priorities.

**Old framing (less defensible):** "CORAL generalizes across reasoning task types via crystallization." Tested by Sudoku→ARC transfer.

**New framing (cleaner, more defensible):** *"CORAL is the first reasoning architecture with within-task compounding efficiency — the more instances of a given task it sees, the better its codebook representations become, producing sub-linear marginal cost per inference over time."*

Why this is better:
1. Maps directly onto enterprise unit economics (deploy-then-improve story)
2. Empirically testable on Sudoku alone — no cross-task experiment required
3. Differentiates from LLMs on an axis where LLMs are structurally weak (they don't get cheaper or better with use in a given enterprise context)
4. Doesn't require beating frontier labs at general reasoning

**This changes what we should run next.** The cross-task transfer experiment becomes lower priority and gets reframed as architecture-generality validation rather than mechanism-validation.

---

## Priority order for next session

1. **Multi-seed campaign — `phase3c_option_y` AND `satisfied-owl`, 3-5 seeds each.** Both, not just phase3c. Without satisfied-owl seed variance you can't claim phase3c is meaningfully better. Provision once, run all seeds in batched sequence. Estimated ~$15-25 GPU. Required for arxiv credibility regardless of the comparison story.

2. **Validate CC's torch.compile fix on a short GPU benchmark *before* committing to multi-seed.** Branch `moe-lb-specialization-compile-fix` (NOT MERGED). Approach A — `@torch.compiler.disable(recursive=False)` on `CoralV3ACT.forward`, plus `scripts/train.py` compiles only `H_level`/`L_level` sub-modules. CPU smoke test passes (160 post-consolidation steps, 0 recompile warnings) but has two known gaps: (a) no eval pass in the smoke test, (b) no GPU throughput benchmark. Run a 10-epoch / eval_interval=5 mini-launch first (~$0.20) to verify throughput is ≥6 it/s post-consolidation AND eval completes in <5 min. If both clean, merge and use for multi-seed.

3. **Within-Sudoku compounding curve experiment.** This is the central commercial-thesis test. Train Phase 3c configuration on Sudoku-Extreme subsets of size 100, 1000, 10000, 100000 puzzles. Plot accuracy vs. training set size. Compare slope to Phase 3a control (no codebook) on the same subsets. If the codebook genuinely compounds, the codebook curve should *steepen* with more data while the control flattens. **This is the experiment that anchors the fundraise pitch.** Should arguably come before ARC-AGI work.

4. **ARC-AGI-1 from scratch (architecture-generality validation).** Train a fresh CORAL on ARC, not transfer from sudoku checkpoint. Purpose: prove the architecture isn't sudoku-specific. The medium claim, not the strong one. Don't claim transfer or cross-task efficiency. Should come *after* the compounding curve result, since a positive compounding result makes ARC validation supportive rather than necessary.

5. **Late-run collapse diagnosis** (Phase 3a control collapsed at step 52080 to 30%, Phase 3c didn't — config-specific). Lower priority but worth understanding for paper defensibility.

**Drop further L_lb formulation experiments.** The variant space is large and we don't yet have evidence that exploring it is more valuable than nailing down what we have.

---

## Pending CC task — ARC adapter investigation and design

Two-phase CC task drafted in the previous session, ready to fire:

**Phase 1 — Investigation only.** Study `sapientinc/HRM` ARC-AGI integration. Document tokenization, variable-grid handling, demonstration-pair representation, max seq len, loss formulation, augmentations. Output: `docs/hrm_arc_integration_notes.md`.

**Phase 2 — Design only.** Write `docs/CORAL_v3_ARC_Adapter_Design.md` covering required CORAL config changes, new code needed, what's reusable, **whether K=32 spatial codebook makes sense for 30×30 ARC grids or needs reformulation** (real design question, must be addressed not assumed), and estimated scope.

Branch: `arc-adapter-design` off `dbf21ea`. No implementation. Human review checkpoint after Phase 2.

The full prompt for CC was drafted in the previous session conversation. If not retained, can be regenerated. The two-phase split is deliberate — investigation is independently valuable, and the design doc creates a decision point before committing to implementation.

---

## Open technical debt and side questions

**torch.compile fix validation.** As above. Smoke test passes but has gaps. Mini-benchmark on next Vast run before trusting it for a 2hr campaign.

**Late-run collapse forensics.** Phase 3a control collapsed at step 52080 to 30.56%. Phase 3c at the same step was at 67.62% — clean. Suggests the collapse is config-specific (precision blowup hypothesis from earlier diagnosis). Not blocking but worth investigating before the multi-seed campaign so we know whether to expect similar collapses in seeds.

**CORAL v3 architecture diagram.** Open from previous session — Anwar flagged that the diagram I drew showing input injection to both H and L modules simultaneously is likely incorrect. Deferred until we can read `CoralV3Inner.forward` directly. CC could resolve this with a 5-minute investigation.

**Compounding curve experiment design.** The high-level concept is in this handoff but the experiment isn't designed yet. Open questions before launching:
- Which subsets exactly? 100/1k/10k/100k seems right but should verify these are within Sudoku-Extreme-1K's available range or require subsampling logic
- How many seeds per data scale? Probably 3 minimum, given the seed variance we just learned about
- What's the right comparison config for the control? Phase 3a control or a fresh equivalent?
- Total GPU budget estimate — likely $30-60 if done thoroughly

---

## Strategic context

**Fundraise framing impact.** The 67.62% number is real progress and the mechanism findings are non-trivial, but the *narrative anchor* for the raise should be the within-Sudoku compounding curve, not the headline accuracy. Reasoning: a single-seed 1.57pp improvement is fragile in due-diligence conversations ("what's the seed variance?" → "we ran one seed" is a credibility hit). A demonstrated *compounding curve* is much harder to dismiss because it's a structural claim about the architecture rather than a point estimate.

**Multi-seed must land before any pitch slides claim 67.62% as "the number."** This is a 1-2 week experimental commitment that has to fit before pitch materials go out.

**Competitive positioning unchanged.** Within-task compounding is a clean differentiator vs. GRAM, Kona, Augmented HRM, frontier LLMs. None of them have this story. Phase 3c didn't change the competitive landscape — it provided the first concrete evidence that CORAL's mechanism can produce the kind of structured codebook required for the compounding claim to be testable.

---

## Process notes (for Claude reading this)

**Oscillation discipline rule, reinforced.** During the Phase 3c run I updated my interpretation of the result 5 times based on intermediate metrics arriving between evals (broken → working → collapsed → converging-to-uniform → sparse-specialization). The final result settled it but my pre-step-52k call ("63-65% expected, parity-to-slightly-worse") was wrong by 2-3pp. Anwar consistently held the line — "let's wait for the next eval" — and was correct every time.

**Discipline rule:** stop interpreting between evals. Require **two consecutive evals in the same direction** before treating a trajectory shift as real. When tempted to update interpretation between data points, the right move is to wait, not to interpret intermediate signals.

**Division of labor unchanged.** CC handles all implementation, tests, commits. Claude handles experiment interpretation, architecture decisions, kill/continue calls, and strategic framing. Anwar pastes terminal output for live interpretation and provides the executive correction when Claude over-interprets.

---

## Resumption instructions for next session

1. Read this handoff.
2. Confirm with Anwar what state things are in:
   - Has the ARC adapter CC task been launched? Has it returned investigation/design docs?
   - Has the torch.compile fix been merged or still on its branch?
   - Has multi-seed been started? Has the compounding curve experiment been scoped?
3. If nothing has progressed: the next concrete action is launching the torch.compile fix mini-benchmark on Vast (~$0.20) followed by the multi-seed campaign. The compounding curve experiment can be designed in parallel without GPU.
4. If CC has returned ARC adapter design docs: review them with Anwar, decide whether ARC implementation is worth scheduling now or after compounding curve.

**Default if uncertain:** ask Anwar rather than guess.
