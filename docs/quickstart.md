# Quickstart

Two minimal recipes that exercise the certifier and the rollout
simulator. Both assume the package has been installed
([`docs/installation.md`](installation.md)).

## Bounded mode

```python
import numpy as np
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.delay.bounded import BoundedDelayModel
from slackcertify.repair import slack_certify
from slackcertify.simulate.rollout import monte_carlo_rollout

plan = Plan.from_paths(
    agents=[Agent(id=0, start=(0, 0), goal=(2, 0))],
    paths=[Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])],
)
pi_prime, cert = slack_certify(plan, mode="bounded", delta=2)
result = monte_carlo_rollout(
    pi_prime, BoundedDelayModel(delta=2), K=500,
    rng=np.random.default_rng(0),
)
print(f"success_rate={result.success_rate:.3f}  "
      f"total_waits={cert.total_wait_inserted}")
```

## Probabilistic mode

```python
import numpy as np
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.delay.bernoulli import BernoulliDelayModel
from slackcertify.repair import slack_certify
from slackcertify.simulate.rollout import monte_carlo_rollout

plan = Plan.from_paths(
    agents=[Agent(id=0, start=(0, 0), goal=(2, 0))],
    paths=[Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])],
)
# delta=1 aligns the certifier's collision predicate with the
# simulator's any-integer-tick occupation-window overlap. See
# docs/algorithm.md §"Aligning with the simulator's collision model"
# for the semantic justification — calling slack_certify without
# delta defaults to 0 (point coincidence only) and under-estimates
# realised collisions on plans where a waiting agent occupies its
# cell for more than one tick.
pi_prime, cert = slack_certify(
    plan, mode="probabilistic", p_d=0.03, epsilon=0.05, delta=1,
)
result = monte_carlo_rollout(
    pi_prime, BernoulliDelayModel(p_d=0.03), K=500,
    rng=np.random.default_rng(0),
)
print(f"success_rate={result.success_rate:.3f}  "
      f"total_waits={cert.total_wait_inserted}")
```

## End-to-end

For the full smoke that exercises every Phase 7 runner and the
Phase 8 analysis pipeline against the bundled fixtures, run:

```bash
bash scripts/repro_smoke.sh
```

That target completes in under 15 minutes on a stock laptop and
produces every figure (`results/figures_smoke/*.pdf`) and table
(`results/summary_smoke/*.tex`) end-to-end.
