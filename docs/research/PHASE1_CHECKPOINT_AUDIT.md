# Phase 1 checkpoint audit

**VERIFIED (user-provided server reports):** both tasks passed
PASS_LEARNER_RESUME_ONLY in a fresh process. Each restored transitions=100,
updates=1, episode_index=2 and executed exactly one further batch-256 update,
matching the reference next loss. This is not full training or simulator restore.

| Task | Preflight envelope SHA256 | Fresh update seconds |
|---|---|---:|
| PushCube-v1 | 5c06a8f0e7d6e973b132b38d055068aeefb9ac0ed1b1f585afe6f135430c3b78 | 1.56734 |
| PickCube-v1 | 0e83eaaff29574b6d49df432a8895c81c3f34f35239966889fb7b1613f938180 | 1.57870 |

**UNRESOLVED:** the new collector check reuses the actual training collector,
collects a warm episode, saves an envelope at a boundary, then compares the next
reference episode to a fresh process loading that envelope. Observations,
actions, rewards and flags must agree at absolute tolerance 1e-6; counters must
match exactly. It also exercises the full runner's frozen evaluator on two seeds.
Maximum 250 simulation transitions/task, zero optimizer updates. This is a
compatibility check, not a training pilot. Any mismatch fails the launch gate.
No general bitwise GPU determinism claim follows from a passing check.

The gate report binds code/config SHA256 and the original preflight envelope hash.
Full-run checkpoint metadata records SHA256, counters and replay device; envelope
provenance binds resolved configuration, sources, dependency lock and packages.
Only trusted local checkpoint files may be loaded (torch weights_only=False).

**Implemented, runtime UNRESOLVED:** interval/full-run envelope saving and resume,
separate latest/final, validation best-model and best resumable envelope where
available. If best is the initial model, best-model.pt is its exact artifact and
there is no resumable best.pt. Metadata/hash mismatch fails closed, including a
crash between checkpoint publication and sidecar publication. Use a separately
verified interval snapshot only through an explicit recovery procedure; do not
silently substitute checkpoints. Phase 0 artifacts remain untouched.
