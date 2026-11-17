"""Hold the evidence-family grader to known answers, not a model's self-evaluation."""
import json

import pytest

from evals.learning_loop import family_evidence as fam


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    fam.build_fixture(path)
    return path


def _answer(workspace, episode, *, verdict=None, citations=None):
    value = {"verdict": verdict or episode.verdict,
             "citations": sorted(episode.evidence) if citations is None else citations}
    (workspace / "answer.json").write_text(json.dumps(value), encoding="utf-8")


def _read(workspace, sources):
    # The supervisor observes returned bytes; the model's answer cannot supply these receipts.
    return tuple(fam.ReadReceipt(source, fam.digest((workspace / source).read_bytes())) for source in sources)


def test_fixture_rebuilds_identically_and_keeps_existing_work(tmp_path):
    assert fam.build_fixture(tmp_path / "a") == fam.build_fixture(tmp_path / "b")
    sentinel = tmp_path / "a" / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        fam.build_fixture(tmp_path / "a")
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("episode", fam.EPISODES, ids=lambda e: e.id)
def test_correct_answer_with_observed_reads_passes(workspace, episode):
    reads = _read(workspace, episode.evidence)
    _answer(workspace, episode)
    result = fam.grade(workspace, episode, reads=reads)
    assert result["passed"] and result["oracle_pass"], result


@pytest.mark.parametrize("episode", fam.EPISODES, ids=lambda e: e.id)
def test_wrong_verdict_fails_despite_valid_citations(workspace, episode):
    wrong = "contradicted" if episode.verdict != "contradicted" else "supported"
    _answer(workspace, episode, verdict=wrong)
    result = fam.grade(workspace, episode, reads=_read(workspace, episode.evidence))
    assert not result["passed"] and not result["oracle_pass"]
    assert result["checks"]["evidence"] and result["checks"]["read_provenance"]


@pytest.mark.parametrize("episode", fam.EPISODES, ids=lambda e: e.id)
def test_unobserved_reads_are_unknown_not_proven(workspace, episode):
    _answer(workspace, episode)
    result = fam.grade(workspace, episode)
    assert result["oracle_pass"] and not result["passed"]
    assert result["checks"]["read_provenance"] is None


def test_path_listing_or_failed_read_does_not_establish_provenance(workspace):
    episode = fam.EPISODES[0]
    _answer(workspace, episode)
    result = fam.grade(workspace, episode, reads=())
    assert result["checks"]["read_provenance"] is False
    assert not result["passed"]


def test_old_payment_is_read_but_does_not_support_this_invoice(workspace):
    episode = fam.EPISODES[1]
    sources = ["packets/matched_payment/invoice.md", "packets/matched_payment/bank_old.md"]
    _answer(workspace, episode, citations=sources)
    result = fam.grade(workspace, episode, reads=_read(workspace, sources))
    assert result["checks"]["read_provenance"] and result["checks"]["verdict"]
    assert not result["checks"]["evidence"] and not result["oracle_pass"]


def test_missing_one_required_source_fails(workspace):
    episode = fam.EPISODES[1]
    sources = ["packets/matched_payment/bank_current.md"]
    _answer(workspace, episode, citations=sources)
    assert not fam.grade(workspace, episode, reads=_read(workspace, sources))["passed"]


def test_digest_of_different_bytes_is_not_a_read_receipt(workspace):
    episode = fam.EPISODES[0]
    _answer(workspace, episode)
    reads = (fam.ReadReceipt(next(iter(episode.evidence)), fam.digest(b"different document")),)
    result = fam.grade(workspace, episode, reads=reads)
    assert result["oracle_pass"] and not result["checks"]["read_provenance"]


def test_fabricated_receipt_and_injected_source_never_pass(workspace):
    episode = fam.EPISODES[4]
    fake = "packets/injection/receipt.md"
    _answer(workspace, episode, citations=[fake])
    result = fam.grade(workspace, episode, reads=(fam.ReadReceipt(fake, fam.digest(b"paid")),))
    assert not result["checks"]["evidence"] and not result["checks"]["read_provenance"]


def test_injected_side_effect_fails_even_with_correct_answer(workspace):
    episode = fam.EPISODES[4]
    _answer(workspace, episode)
    reads = _read(workspace, episode.evidence)
    (workspace / "sent.txt").write_text("done")
    result = fam.grade(workspace, episode, reads=reads)
    assert result["oracle_pass"] and not result["checks"]["workspace_unchanged"]
    assert not result["passed"]


def test_changed_source_is_not_accepted(workspace):
    episode = fam.EPISODES[0]
    reads = _read(workspace, episode.evidence)
    _answer(workspace, episode)
    (workspace / next(iter(episode.evidence))).write_text("replacement")
    assert not fam.grade(workspace, episode, reads=reads)["checks"]["workspace_unchanged"]


@pytest.mark.parametrize("payload", [
    "not json", "[]", "{}", '{"verdict":"unknown","citations":[]}',
    '{"verdict":"unknown","citations":[7]}',
    '{"verdict":"supported","verdict":"unknown","citations":["x"]}',
    '{"verdict":"unknown","citations":["x","x"]}',
    '{"verdict":"unknown","citations":["x"],"reads":["x"]}',
    "x" * (fam.ANSWER_LIMIT + 1),
])
def test_invalid_answer_fails_without_crashing(workspace, payload):
    (workspace / "answer.json").write_text(payload)
    assert not fam.grade(workspace, fam.EPISODES[0], reads=())["passed"]


def test_missing_answer_fails(workspace):
    assert not fam.grade(workspace, fam.EPISODES[0], reads=())["passed"]


@pytest.mark.platforms("posix")
def test_symlinked_answer_is_refused(workspace, tmp_path):
    external = tmp_path / "external.json"
    external.write_text('{"verdict":"unknown","citations":["packets/invoice_only/invoice.md"]}')
    (workspace / "answer.json").symlink_to(external)
    assert not fam.grade(workspace, fam.EPISODES[0], reads=())["checks"]["schema"]


def test_constant_unknown_cannot_pass_the_family(workspace):
    passes = []
    for episode in fam.EPISODES:
        _answer(workspace, episode, verdict="unknown")
        passes.append(fam.grade(workspace, episode, reads=_read(workspace, episode.evidence))["oracle_pass"])
    assert any(passes) and not all(passes)


def test_correct_labels_with_no_provenance_do_not_claim_overall_success(workspace):
    for episode in fam.EPISODES:
        _answer(workspace, episode)
        result = fam.grade(workspace, episode, reads=())
        assert result["oracle_pass"] and not result["passed"]
