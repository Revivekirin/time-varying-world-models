# Research scope

Online Identification and Decision-Relevant Adaptation to Hidden Time-Varying
Dynamics in Latent World Models.

System: x[t+1] = F(x[t], a[t]; μ[t]); o[t] = H(x[t]); μ[t+1] = G(μ[t], ξ[t]).
The future hypothesis c[t] = gφ(H[t]), z[t] = hθ(o[t]),
z[t+1] = dθ(z[t],a[t],c[t]) is not implemented or established in Phase 0.
History must be causal and exclude privileged or future dynamics values.

Preserved roadmap:
0. Integration, environment contracts, bounded learner smoke.
1. Nominal TD-MPC2 training and independent evaluation on both tasks.
2. Intervention validity and severity calibration.
3. Frozen nominal model under fixed, abrupt, gradual, reversible shifts.
4. Matched dynamics-randomized versus history-conditioned TD-MPC2 plus oracle.
5. History information → conditional prediction → action ranking → closed-loop control.
6. Optional online neural-parameter adaptation.

Five independent claims: (1) system operates; (2) nominal policy solves tasks;
(3) intervention changes actual transitions; (4) history adds useful information;
(5) history conditioning improves decisions/control. No claim implies the next.

Later models: M0 nominal; M1 randomized/current observation; M2 randomized/history;
M3 privileged oracle; M4 optional online parameter updates. M1 versus M2 must match
data distributions, interactions, optimizer budgets and evaluation. Oracle is a
reference, not a guarantee of optimal control.

Only state observations and one shared TD-MPC2 integration are in scope now.
Original TD-MPC is source comparison only. No full training, history network,
shift implementation/rollout, domain-randomized training, tuning sweep, SAC,
VLA/RLVR/PPO/GRPO, or previous repository changes are authorized.
Native task reset randomization remains native; no added dynamics randomization.
The user exclusively executes all remote commands. Phase 1 pilot and full run
require separate decisions; Phase 2 requires a competent nominal checkpoint.
