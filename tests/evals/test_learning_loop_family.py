"""The "conventions" family's oracle, held to known answers: it must accept correct work on every episode and fail
each kind of mistake on exactly the check that names it. A grader that passed everything, or nothing, would make
any learning-loop result meaningless, so both directions are pinned here, with real git and real subprocesses.
"""
import subprocess

import pytest

from evals.learning_loop import family_conventions as fam

EPISODES = {e.id: e for e in fam.EPISODES}


def _edit(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {path}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _log(ws, line):
    _edit(ws / "CHANGES.md", "## Unreleased\n", f"## Unreleased\n\n{line}\n")


def _commit(ws, message):
    subprocess.run(["git", "-C", str(ws), "add", "-A"], check=True)
    return subprocess.run(["git", "-C", str(ws), "commit", "-q", "-m", message], capture_output=True, text=True,
                          check=False)


def _percent_of(ws):
    with (ws / "invoicer/money.py").open("a", encoding="utf-8") as f:
        f.write('\n\ndef percent_of(amount: Decimal, percent: Decimal) -> Decimal:\n'
                '    return (amount * percent / Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)\n')


def _skip_empty_lines(ws):
    _edit(ws / "invoicer/invoice.py", "for item in self.items)", "for item in self.items if item.quantity > 0)")


def _currency_symbol(ws):
    _edit(ws / "invoicer/money.py", "def format_amount(amount: Decimal) -> str:",
          "def format_amount(amount: Decimal, currency: str | None = None) -> str:")
    _edit(ws / "invoicer/money.py", 'return f"{quantized:,.2f}"',
          'return {"EUR": "€", "USD": "$"}.get(currency, "") + f"{quantized:,.2f}"')


def _discounted_total(ws):
    _edit(ws / "invoicer/invoice.py", "from decimal import Decimal", "from decimal import ROUND_HALF_UP, Decimal")
    with (ws / "invoicer/invoice.py").open("a", encoding="utf-8") as f:
        f.write('\n    def discounted_total(self, percent: Decimal) -> Decimal:\n'
                '        discounted = self.subtotal() * (1 - percent / Decimal("100"))\n'
                '        return (discounted * (1 + self.tax_rate)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)\n')


def _parse_symbols(ws):
    _edit(ws / "invoicer/money.py", 'text.replace(",", "").strip()', 'text.replace(",", "").strip().lstrip("€$£")')


SOLUTIONS = {"percent_of": _percent_of, "skip_empty_lines": _skip_empty_lines, "currency_symbol": _currency_symbol,
             "discounted_total": _discounted_total, "parse_symbols": _parse_symbols}


def _solve(ws, episode_id, *, log=True, area=None, commit=True):
    """Apply a correct change for the episode, then (optionally) log it and commit it."""
    episode = EPISODES[episode_id]
    SOLUTIONS[episode_id](ws)
    summary = f"handle {episode_id.replace('_', ' ')}"
    if log:
        _log(ws, f"- {episode.area}: {summary}")
    if commit:
        result = _commit(ws, f"{area or episode.area}: {summary}")
        assert result.returncode == 0, result.stderr


@pytest.fixture
def ws(tmp_path):
    workspace = tmp_path / "ws"
    return workspace, fam.build_fixture(workspace)


def test_fixture_is_byte_identical_across_builds(tmp_path):
    assert fam.build_fixture(tmp_path / "a") == fam.build_fixture(tmp_path / "b")


def test_tests_only_run_through_the_project_script(ws):
    workspace, _ = ws
    bare = subprocess.run(["python3", "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=workspace,
                          capture_output=True, text=True, check=False)
    assert bare.returncode != 0 and "APP_MODE is not set" in bare.stderr
    assert subprocess.run(["./scripts/check"], cwd=workspace, capture_output=True, check=False).returncode == 0


def test_commit_hook_rejects_a_message_outside_the_rule(ws):
    workspace, _ = ws
    (workspace / "README.md").write_text("changed\n", encoding="utf-8")
    rejected = _commit(workspace, "Fix readme wording")
    assert rejected.returncode != 0 and "<area>: <summary>" in rejected.stderr
    assert _commit(workspace, "docs: fix the readme wording").returncode == 0


@pytest.mark.parametrize("episode_id", sorted(EPISODES))
def test_correct_work_passes_every_check(ws, episode_id):
    workspace, base = ws
    _solve(workspace, episode_id)
    result = fam.grade(workspace, EPISODES[episode_id], base)
    assert result["passed"], result["checks"]


def test_untouched_project_earns_nothing_but_a_green_suite(ws):
    workspace, base = ws
    result = fam.grade(workspace, EPISODES["percent_of"], base)
    assert result["checks"] == {"hidden_tests": False, "suite_green": True, "commit": False, "changelog": False}


def test_missing_changelog_line_fails_only_the_changelog_check(ws):
    workspace, base = ws
    _solve(workspace, "percent_of", log=False)
    assert fam.grade(workspace, EPISODES["percent_of"], base)["checks"] == {
        "hidden_tests": True, "suite_green": True, "commit": True, "changelog": False}


def test_commit_under_the_wrong_area_fails_the_commit_check(ws):
    workspace, base = ws
    _solve(workspace, "percent_of", area="invoice")
    checks = fam.grade(workspace, EPISODES["percent_of"], base)["checks"]
    assert checks["commit"] is False and checks["hidden_tests"] is True


def test_uncommitted_work_fails_the_commit_check(ws):
    workspace, base = ws
    _solve(workspace, "percent_of", commit=False)
    checks = fam.grade(workspace, EPISODES["percent_of"], base)["checks"]
    assert checks["commit"] is False and checks["hidden_tests"] is True and checks["changelog"] is True
