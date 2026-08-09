"""Hidden tests — speculative_decoding_verify.

Every assertion is licensed by a sentence in prompt.md, and every promise in
prompt.md has a test here that enforces it.

The task's whole claim is a statement about a distribution, so most of this file
is one simulation per scenario and a set of exact expectations computed from the
inputs alone. Nothing here compares against the reference solution: the target
distribution, the acceptance probability `sum(min(p, q))` and the survival
probabilities `prod(alpha)` are all closed-form consequences of the
specification, so a mistake shared between the reference and the tests has
nowhere to hide.

Tolerances. A frequency measured over `n` independent trials has standard error
at most `sqrt(0.25 / n)`, which is 0.0011 at 200000 trials, so `atol=0.008` is
a six-sigma band and `atol=0.012` is six sigma for the conditional counts,
whose denominators are smaller. A correct implementation clears them with room to
spare; the wrong versions this task exists to catch are off by 0.05 to 0.2,
because they emit a different distribution rather than a noisier one.

The simulations are memoised so that the twenty-odd tests reading them cost one
pass each rather than one pass per test. They are called from inside the test
bodies rather than from a fixture, so an unimplemented solution fails the tests
instead of erroring in setup.
"""

import numpy as np
import pytest

from speculative_decoding import verify_draft


# -- scenarios -------------------------------------------------------------
#
# `divergent` has the two models disagreeing hard, so roughly half the
# proposals are rejected and the correction term carries most of the output
# distribution. `aligned` has them nearly agreeing, so almost every round runs
# to the end and the free token at the tail is well sampled.

DIVERGENT = {
    "name": "divergent",
    "draft": np.array(
        [
            [0.05, 0.10, 0.20, 0.25, 0.40],
            [0.30, 0.10, 0.20, 0.20, 0.20],
            [0.20, 0.30, 0.20, 0.20, 0.10],
        ]
    ),
    "target": np.array(
        [
            [0.40, 0.25, 0.20, 0.10, 0.05],
            [0.10, 0.50, 0.15, 0.15, 0.10],
            [0.25, 0.25, 0.25, 0.15, 0.10],
            [0.30, 0.05, 0.35, 0.20, 0.10],
        ]
    ),
    "trials": 200_000,
    "seed": 11,
}

ALIGNED = {
    "name": "aligned",
    "draft": np.array(
        [
            [0.32, 0.28, 0.22, 0.18],
            [0.12, 0.38, 0.36, 0.14],
            [0.45, 0.25, 0.20, 0.10],
        ]
    ),
    "target": np.array(
        [
            [0.30, 0.30, 0.20, 0.20],
            [0.10, 0.40, 0.40, 0.10],
            [0.50, 0.20, 0.20, 0.10],
            [0.25, 0.25, 0.25, 0.25],
        ]
    ),
    "trials": 200_000,
    "seed": 23,
}

SCENARIOS = {scenario["name"]: scenario for scenario in (DIVERGENT, ALIGNED)}


# -- expectations computed from the inputs, not from the solution ----------


def acceptance_probabilities(scenario):
    """`sum(min(p, q))` per position: the largest rate the guarantee allows."""
    draft, target = scenario["draft"], scenario["target"]
    return np.minimum(draft, target[: len(draft)]).sum(axis=1)


def survival_probabilities(scenario):
    """`P(at least k proposals accepted)` for k = 0 .. n_draft."""
    return np.concatenate([[1.0], np.cumprod(acceptance_probabilities(scenario))])


# -- the simulation --------------------------------------------------------

_RESULTS = {}


def simulate(name):
    if name not in _RESULTS:
        _RESULTS[name] = _run(SCENARIOS[name])
    return _RESULTS[name]


def _run(scenario):
    draft, target = scenario["draft"], scenario["target"]
    n_draft, vocab = draft.shape
    trials = scenario["trials"]

    # The proposals are drawn from the draft distributions, vectorised and up
    # front, so that every implementation sees exactly the same proposals and
    # the generator the solution draws from is a different one entirely. Two
    # streams, so how much randomness a solution consumes cannot move the
    # inputs it is handed.
    token_rng = np.random.default_rng(scenario["seed"])
    proposals = np.stack(
        [token_rng.choice(vocab, size=trials, p=draft[position]) for position in range(n_draft)],
        axis=1,
    )
    solver_rng = np.random.default_rng(scenario["seed"] + 1)

    first = np.zeros(vocab, dtype=np.int64)
    second = np.zeros(vocab, dtype=np.int64)
    tail = np.zeros(vocab, dtype=np.int64)
    accepted = np.zeros(n_draft + 1, dtype=np.int64)
    bad_length = 0
    outside_vocabulary = 0
    prefix_broken = 0
    rejected_token_returned = 0

    for trial in range(trials):
        proposal = proposals[trial]
        emitted = np.asarray(verify_draft(proposal, draft, target, solver_rng))

        if emitted.ndim != 1 or not 1 <= emitted.shape[0] <= n_draft + 1:
            bad_length += 1
            continue
        if emitted.size and not np.all((emitted >= 0) & (emitted < vocab)):
            outside_vocabulary += 1
            continue

        n_accepted = emitted.shape[0] - 1
        accepted[n_accepted] += 1
        if not np.array_equal(emitted[:n_accepted], proposal[:n_accepted]):
            prefix_broken += 1
        if n_accepted < n_draft and emitted[-1] == proposal[n_accepted]:
            rejected_token_returned += 1

        first[emitted[0]] += 1
        if n_accepted >= 1:
            second[emitted[1]] += 1
        if n_accepted == n_draft:
            tail[emitted[-1]] += 1

    return {
        "trials": trials,
        "first": first,
        "second": second,
        "tail": tail,
        "accepted": accepted,
        "bad_length": bad_length,
        "outside_vocabulary": outside_vocabulary,
        "prefix_broken": prefix_broken,
        "rejected_token_returned": rejected_token_returned,
    }


# -- the guarantee: the emitted token is distributed as the target ---------


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_first_emitted_token_is_distributed_as_the_target(name):
    scenario = SCENARIOS[name]
    result = simulate(name)
    frequency = result["first"] / result["trials"]
    # Six standard errors at 200000 trials. An implementation that resamples
    # from the target on rejection, or renormalises the wrong residual, lands
    # 0.05 to 0.2 away here rather than 0.002.
    assert np.allclose(frequency, scenario["target"][0], atol=0.008, rtol=0.0)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_second_emitted_token_is_distributed_as_the_target(name):
    # Conditioned on the first proposal surviving, the second position is an
    # independent round of the same experiment, so its emitted token has to be
    # the target's second distribution exactly.
    scenario = SCENARIOS[name]
    result = simulate(name)
    reached = result["second"].sum()
    assert reached > 1000, "too few rounds reached the second position to measure it"
    # Six standard errors at the smallest denominator this branch produces
    # (roughly 100000 rounds in the divergent scenario).
    assert np.allclose(
        result["second"] / reached, scenario["target"][1], atol=0.012, rtol=0.0
    )


def test_the_free_token_comes_from_the_position_after_the_proposal():
    # `aligned` accepts all three proposals about 86% of the time, so the tail
    # distribution is measured over roughly 171000 rounds.
    scenario = ALIGNED
    result = simulate("aligned")
    complete = result["tail"].sum()
    assert complete > 1000, "too few rounds ran to the end to measure the free token"
    assert np.allclose(
        result["tail"] / complete, scenario["target"][-1], atol=0.012, rtol=0.0
    )


# -- the guarantee: acceptance is as high as it can be --------------------


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_acceptance_rate_is_the_largest_the_guarantee_allows(name):
    scenario = SCENARIOS[name]
    result = simulate(name)
    expected = survival_probabilities(scenario)
    measured = np.cumsum(result["accepted"][::-1])[::-1] / result["trials"]
    # Same six-sigma band as the token frequencies: these are proportions over
    # the same 200000 trials. A scheme that accepts less often than
    # `min(1, p/q)` is a slower decoder emitting the right distribution, and
    # this is the only thing that catches it.
    assert np.allclose(measured, expected, atol=0.008, rtol=0.0)


# -- the shape of a round --------------------------------------------------


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_round_emits_between_one_token_and_one_more_than_it_was_offered(name):
    assert simulate(name)["bad_length"] == 0


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_emitted_token_is_in_the_vocabulary(name):
    assert simulate(name)["outside_vocabulary"] == 0


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_everything_before_the_last_emitted_token_is_the_proposal_itself(name):
    assert simulate(name)["prefix_broken"] == 0


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_a_rejected_proposal_is_never_the_token_that_replaces_it(name):
    # Rejection only happens where the target wanted less of the token than the
    # draft offered, and the correction is the part the target wanted *more*
    # of, which is zero there. So the replacement can never be the token that
    # was just turned down.
    assert simulate(name)["rejected_token_returned"] == 0


# -- cases with no randomness left in them --------------------------------


def test_two_identical_models_accept_everything():
    # p equals q, so the ratio is one everywhere and the largest permitted
    # acceptance probability is certainty.
    probabilities = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.25, 0.25, 0.5]])
    proposal = np.array([2, 1])
    rng = np.random.default_rng(5)

    for _ in range(200):
        emitted = np.asarray(verify_draft(proposal, probabilities[:2], probabilities, rng))
        assert emitted.shape == (3,)
        assert np.array_equal(emitted[:2], proposal)


def test_a_proposal_the_target_gives_no_mass_to_is_always_rejected():
    draft = np.array([[1.0, 0.0, 0.0, 0.0]])
    target = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    rng = np.random.default_rng(6)

    for _ in range(200):
        emitted = np.asarray(verify_draft(np.array([0]), draft, target, rng))
        # One token out, and the only token the target wanted more of than the
        # draft offered is the third.
        assert emitted.shape == (1,)
        assert emitted[0] == 2


def test_an_empty_proposal_still_emits_the_free_token():
    target = np.array([[0.0, 0.0, 0.0, 1.0]])
    draft = np.zeros((0, 4))
    rng = np.random.default_rng(7)

    emitted = np.asarray(verify_draft(np.array([], dtype=np.int64), draft, target, rng))
    assert emitted.shape == (1,)
    assert emitted[0] == 3


def test_a_proposal_the_draft_and_target_both_love_survives_to_the_free_token():
    draft = np.array([[0.9, 0.1]])
    target = np.array([[0.95, 0.05], [0.0, 1.0]])
    rng = np.random.default_rng(8)

    accepted = 0
    for _ in range(500):
        emitted = np.asarray(verify_draft(np.array([0]), draft, target, rng))
        if len(emitted) == 2:
            accepted += 1
            assert emitted[0] == 0
            assert emitted[1] == 1
    # The target wants token 0 more than the draft offered it, so acceptance is
    # certain and every one of the 500 rounds runs to the free token.
    assert accepted == 500


# -- interface -------------------------------------------------------------


def test_the_result_is_a_one_dimensional_array_of_token_ids():
    draft = np.array([[0.4, 0.6], [0.5, 0.5]])
    target = np.array([[0.5, 0.5], [0.6, 0.4], [0.3, 0.7]])
    emitted = verify_draft(np.array([0, 1]), draft, target, np.random.default_rng(9))
    emitted = np.asarray(emitted)
    assert emitted.ndim == 1
    assert np.issubdtype(emitted.dtype, np.integer)


def test_float32_probabilities_are_accepted():
    draft = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
    target = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    rng = np.random.default_rng(10)

    for _ in range(100):
        emitted = np.asarray(verify_draft(np.array([0]), draft, target, rng))
        # The target gives token 0 no mass, so the round always ends at the
        # correction, and the correction can only be token 3.
        assert emitted.shape == (1,)
        assert emitted[0] == 3


def test_the_inputs_are_left_alone():
    draft = np.array([[0.3, 0.3, 0.4], [0.2, 0.5, 0.3]])
    target = np.array([[0.5, 0.2, 0.3], [0.4, 0.4, 0.2], [0.1, 0.1, 0.8]])
    proposal = np.array([2, 0])
    originals = [draft.copy(), target.copy(), proposal.copy()]

    rng = np.random.default_rng(12)
    for _ in range(100):
        verify_draft(proposal, draft, target, rng)

    for original, current in zip(originals, [draft, target, proposal]):
        assert np.array_equal(original, current)


def test_two_generators_in_the_same_state_produce_the_same_round():
    # The only randomness in a round comes from the generator it was handed, so
    # the function is reproducible when the generator is.
    draft = np.array([[0.4, 0.35, 0.25], [0.3, 0.3, 0.4]])
    target = np.array([[0.2, 0.5, 0.3], [0.45, 0.35, 0.2], [0.3, 0.3, 0.4]])
    proposal = np.array([0, 2])

    one = verify_draft(proposal, draft, target, np.random.default_rng(13))
    other = verify_draft(proposal, draft, target, np.random.default_rng(13))
    assert np.array_equal(np.asarray(one), np.asarray(other))


def test_a_vocabulary_of_one_leaves_nothing_to_disagree_about():
    draft = np.array([[1.0], [1.0]])
    target = np.array([[1.0], [1.0], [1.0]])
    emitted = np.asarray(
        verify_draft(np.array([0, 0]), draft, target, np.random.default_rng(14))
    )
    assert np.array_equal(emitted, [0, 0, 0])
