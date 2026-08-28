# wasp_MARL

A decentralized multi-agent simulation of wasp-colony foraging logistics, applied
to coordinated search and rescue. No agent has a map, a global plan, or a central
scheduler: coverage and rescue emerge from local sensing, pairwise communication,
and stigmergic markers dropped in the environment.

## What the project is about

Paper-wasp colonies distribute food through a nest with no dispatcher. Foragers
return with loads, nurses redistribute them, and the allocation that results is
efficient without anyone computing it. The logistical structure of that problem —
agents that must cover unknown ground, find things, and recruit enough help to
extract them, all without global state — is the same structure as a
Coordinated Search and Rescue (CSAR) mission.

This repo reproduces the wasp feeding dynamics as an agent-based model and then
uses the control algorithms it exposes as a decentralized CSAR controller.

**Agent roles** (`Role` in `agent.py`), assigned at initialisation by
`PlanningService.assign_roles` and reassigned during the run by
`MemoryService.update_roles`:

- `EXPLORER` — pushes outward to an assigned exploration centroid and sweeps it.
- `COMMUNICATOR` — holds position in the network interior, relaying between
  explorers and the nest. Movement is constrained to keep the relay chain intact.
- `RESCUE` — recruited to a found target; a target is only extracted once enough
  rescuers have converged on it.

**Decoys** (`DecoyAgent`) are the stigmergic layer — markers dropped into the
environment rather than messages sent to peers. Each carries a `Status`:
`FOUND` (a target is here, come help), `EXPLORED` (covered, do not re-cover),
`SATURATED` (enough agents already committed here, go elsewhere), `ENEMY`. This is
how the swarm avoids both redundant coverage and over-recruitment without any
agent holding a global picture.

**Targets** (`LostAgent`) are placed on a ring just outside the initial
exploration radius, each requiring a fixed number of rescuers (3) to extract.

The dynamics are stochastic and second-order: agents integrate acceleration under
speed and acceleration caps (`v_max`, `a_max`), damping, and additive noise, with
attractive and repulsive terms toward decoys/targets and away from neighbours
(`movement_services.py`). Memory decays — `forget_seen_neighbors` and
`forget_stored_positions` run on a `forget_frequency` schedule, and the
communication threshold decays each step — so the swarm cannot rely on stale
information indefinitely.

## Layout

The simulation is organised as a set of stateless services applied to mutable
agent dataclasses, one service per concern. `Simulator.step` calls them in a fixed
order each tick:

```
main.py                       loads config, builds agents and targets, runs, dumps JSON
simulator.py                  the tick loop; step() = move -> sense -> memory ->
                              behavior, and records the schedule
agent.py                      Agent, LostAgent, DecoyAgent, Role, Status, and the
                              frozen Position / Speed / Acceleration / Noise types
movement_services.py          all motion: per-role movement estimation, attractive
                              and repulsive terms, stochastic and deterministic
                              integration, speed/role constraint projection
communication_services.py     sensing within radius, message generation
memory_services.py            forgetting, role updates, leader election
planning_services.py          initial role split, exploration centroid assignment
                              and route planning
behavior_service.py           decoy dropping (found / explored) and saturation
config/config.yaml            every parameter
process_data.R                figures from outputs/data.json
outputs/data.json             a committed run, so the plots work without simulating
```

## Configuration

All parameters live in `config/config.yaml` — nothing is passed on the command
line. The two blocks map directly onto `Agent.__init__` and
`Simulator.run_simulation`:

```yaml
agent:
  v_max: 1.5                        # speed cap
  a_max: 0.5                        # acceleration cap
  damp: 10.0
  nest_radius: 3
  kappa: 8.0                        # attraction / repulsion gains
  sigma: 1.5                        # noise scale
  theta: 0.5
  sensing_radius: 1
  communication_threshold: 0
  communication_threshold_decay: 0.9
  exploration_buffer_radius: 2.0
simulator:
  simulation_length: 1000           # ticks
  num_agents: 20
  proportion_explorers: 0.3         # the rest start as communicators
  exploration_radius: 3
  exploration_period: 1000
  forget_frequency: 5               # ticks between memory decay
  num_lost_agents: 2                # rescue targets
  radius_decoys: 6                  # decoy influence radius
```

## Setup and running

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install numpy pyyaml pandas matplotlib torch
```

There is no `requirements.txt`. `torch` is imported by
`movement_services.py`; the rest of the simulation is numpy.

```bash
python main.py
```

Runs from the repository root — `main.py` opens `config/config.yaml` by relative
path. Progress is logged per agent per tick at `INFO` level (role, position,
exploration state), which is verbose for 20 agents x 1000 ticks; redirect it or
raise the level in `Simulator.run_simulation` if it gets in the way.

## Output

`outputs/data.json`, with four keys:

| key | contents |
|---|---|
| `schedule` | the full trajectory record: per tick, a flat list of `[id, x, y, role, found_state]` repeated for every agent — reshape to 5 columns |
| `decoys` | every marker dropped: `id`, `x`, `y`, `role` (the `Status` value), `num_agents` committed |
| `exploration_centroids` | the centroid each explorer was assigned |
| `exploration_centroids_expansion` | per explorer id, the sequence of centroids it swept |

A committed `outputs/data.json` from a previous run is in the repository, so the
plots can be reproduced without simulating.

## Figures

`process_data.R` reads `outputs/data.json` and produces the trajectory, coverage
and kernel-density figures. It needs a fair number of R packages:

```r
install.packages(c("dplyr", "MASS", "ks", "slider", "ggplot2", "RColorBrewer",
                   "ggnewscale", "cowplot", "HDInterval", "zoo", "tidyr",
                   "tibble", "tseries", "forecast", "patchwork", "jsonlite",
                   "ggforce"))
```

**Before running it, fix line 19.** The script hard-codes

```r
setwd("C:/Users/samil/Documents/wasp_MARL/outputs")
```

Point that at your own `outputs/` directory, or delete the line and run the script
with `outputs/` as the working directory.

## Reproducibility

`main.py` calls `np.random.uniform` for target placement without seeding, and the
movement services add unseeded noise every step, so **each run differs**. Set
`np.random.seed()` and `random.seed()` at the top of `main.py` for a repeatable
run.

## Context

Feeds the CSER 2026 submission *Swarm Intelligence for Generalist Agents
Coordination in CSAR Missions* (Cornejo, S., Salado, A.). Listed under Projects as
*Paper-Wasp Feeding Dynamics Simulation*.

Note on the repository name: despite the `_MARL` suffix, there is no
reinforcement learning here. Coordination is rule-based — roles, stigmergic
decoys, and local sensing — with no reward, policy, or training loop. `torch` is
used only for the autograd gradient steps in `movement_services.py`.

Author: Samuel Cornejo (<samuelcornejo@arizona.edu>)
