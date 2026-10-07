"""Playing episodes: one worker process holds its environments and plays whole episodes with the current weights.

The reward is the week's cost divided by what the naive rule spends in an average week of that network, so a week
of Small and a week of Full are worth the same and can be learnt from together (the package's own ``ScaleReward``
table). The scenarios come from a pool drawn once per root and kept on disk, so a reset costs milliseconds.
"""

import os

import numpy as np
import torch


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

RULES_FOLDER = None  # set by `setup`
_STATE = {}


def _env(task, entropy, n_scenarios):
    import gymnasium as gym
    from shockbench_flow_gym.wrappers import ScenarioPool

    from sbf_starter import env_id

    key = (task, entropy, n_scenarios)
    if key not in _STATE:
        base = gym.make(env_id(task), regime="standard")
        _STATE[key] = ScenarioPool(base, n_scenarios, entropy)
    return _STATE[key]


def reward_scale(task):
    from shockbench_flow_gym.wrappers import REWARD_SCALE_USD

    return float(REWARD_SCALE_USD[f"chokepoint-{task}"])


def setup(rules_folder, threads=1):
    global RULES_FOLDER
    RULES_FOLDER = rules_folder
    torch.set_num_threads(threads)


def _policy(version, path, kind, sizes, width, rounds):
    import nets

    cached = _STATE.get("policy")
    if cached is None or cached[0] != version:
        policy = nets.build(kind, sizes, width, rounds)
        policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        policy.eval()
        _STATE["policy"] = (version, policy)
    return _STATE["policy"][1]


def play(job):
    """One episode. Returns the weeks' tables, what was sampled, and the week's scaled cost.

    ``job`` carries everything a worker needs: which network, which scenario, which weights, which feature groups.
    With ``sample=False`` the mean correction is taken and only the costs come back — that is how a variant is
    measured against the plain rules on the very same scenario.
    """
    import gymnasium as gym  # noqa: F401  (imported for the side effect of registering the ids)
    import residual as R
    from shockbench_flow_gym import agent_config_from_reset

    task = job["task"]
    env = _env(task, job["entropy"], job["n_scenarios"])
    obs, info = env.reset(options={"pool_index": int(job["pool_index"])})
    config = agent_config_from_reset(env, obs, info)

    rules = R.load_rules(RULES_FOLDER)
    rules_only = job.get("rules_only", False)
    policy = (
        None
        if rules_only
        else _policy(job["version"], job["policy_path"], job["kind"], job["sizes"], job["width"], job["rounds"])
    )
    agent = R.Residual(
        config, rules, policy, groups=job.get("groups"), lo=job["lo"], hi=job["hi"], families=job.get("families")
    )

    gen = torch.Generator().manual_seed(int(job["seed"]))
    scale = reward_scale(task)
    weeks, costs = [], []
    while True:
        if rules_only:  # the plain rules on this scenario: the control the policy's weeks are measured against
            action = agent.rules_action(obs)
        elif job["sample"]:
            action, record = agent.step(obs, gen)
            if job.get("record", True):  # an evaluation that samples does not need the tables sent back
                weeks.append(record)
        else:
            action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        costs.append(-float(reward) / scale)  # a positive number: what this week cost, in naive weeks
        if terminated or truncated:
            break
    out = {
        "task": task,
        "pool_index": int(job["pool_index"]),
        "costs": np.array(costs, dtype=np.float64),
        "fallbacks": agent.fallbacks,
    }
    if job["sample"] and job.get("record", True):
        out["weeks"] = weeks
    return out


def sizes_for(task, rules_folder, groups=None, entropy=0, n_scenarios=1):
    """The width of each table on ``task``, and its index arrays — what the networks need before they exist."""
    import batching
    import gymnasium as gym
    import residual as R
    from shockbench_flow_gym import agent_config_from_reset

    from sbf_starter import env_id

    env = gym.make(env_id(task), regime="standard", entropy=entropy)
    obs, info = env.reset(seed=0)
    config = agent_config_from_reset(env, obs, info)
    rules = R.load_rules(rules_folder)
    agent = R.Residual(config, rules, None, groups=groups)
    action = agent.rules_action(obs)
    tables = agent.tables(obs, action["flows"])
    env.close()
    return agent.feat.sizes, batching.index_arrays(agent.layout), tables
