"""Task family "conventions": one small project, five changes, a workflow worth learning once.

Every episode is one realistic change to the same invoicing library. Three rules apply to every change and are
written down only in CONTRIBUTING.md, so an agent either reads them (or a skill it learned earlier) each time, or
gets them wrong:

  1. tests run through ./scripts/check, which sets APP_MODE=test; a bare `python3 -m unittest` fails on import;
  2. the commit message is "<area>: <summary>" with area one of money, invoice, docs or tests; the repository's
     commit-msg hook rejects anything else and prints the rule;
  3. CHANGES.md gets "- <area>: <summary>" under "## Unreleased". Nothing enforces this one.

The fixture is rebuilt before every episode, byte-identical (fixed content, identity and dates), so two episodes
differ only in their task and in what the agent carried over. Grading is programmatic and needs no model: hidden
tests, the project's own suite, the commit and the changelog line.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

AREAS = ("money", "invoice", "docs", "tests")
# A fixed identity and clock make the initial commit, and so every episode's starting point, byte-identical.
_FIXTURE_ENV = {"GIT_AUTHOR_NAME": "Invoicer Maintainer", "GIT_AUTHOR_EMAIL": "maintainer@invoicer.test",
                "GIT_COMMITTER_NAME": "Invoicer Maintainer", "GIT_COMMITTER_EMAIL": "maintainer@invoicer.test",
                "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}
# Hidden tests and the project's suite are a handful of unit tests; a minute means a hang, not slowness.
GRADE_TIMEOUT_S = 60

FILES = {
    "README.md": "# invoicer\n\nSmall invoicing helpers: parse and format money amounts, total invoices with tax.\n",
    "CONTRIBUTING.md": """# Contributing to invoicer

Every change follows the same three steps.

1. **Test through `./scripts/check`.** It sets `APP_MODE=test`; the package refuses to import without it, so a
   bare `python3 -m unittest` fails. Add or update a test for every change.
2. **Commit as `<area>: <summary>`.** `<area>` is one of `money`, `invoice`, `docs` or `tests`; the summary is
   lower case, 5 to 70 characters. The commit-msg hook in `.githooks/` rejects anything else.
3. **Log it in `CHANGES.md`.** Add `- <area>: <summary>` as the first line under `## Unreleased`, in the same
   commit.
""",
    "CHANGES.md": "# Changes\n\n## Unreleased\n\n## 1.0.0\n\n- invoice: first release\n",
    ".gitignore": "__pycache__/\n*.pyc\n",
    "invoicer/__init__.py": '"""Small invoicing library."""\n',
    "invoicer/config.py": """import os

MODE = os.environ.get("APP_MODE")
if MODE not in {"test", "prod"}:
    raise RuntimeError("APP_MODE is not set. Run the tests with ./scripts/check (see CONTRIBUTING.md).")
""",
    "invoicer/money.py": '''"""Money helpers. Amounts are Decimal, never float."""
from decimal import ROUND_HALF_UP, Decimal

from invoicer import config  # noqa: F401  (refuses to import outside ./scripts/check)


def parse_amount(text: str) -> Decimal:
    """'1,234.50' -> Decimal('1234.50')."""
    return Decimal(text.replace(",", "").strip())


def format_amount(amount: Decimal) -> str:
    """Decimal('1234.5') -> '1,234.50'."""
    quantized = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{quantized:,.2f}"
''',
    "invoicer/invoice.py": '''"""Invoices: line items and totals."""
from dataclasses import dataclass, field
from decimal import Decimal

from invoicer import config  # noqa: F401


@dataclass
class LineItem:
    description: str
    unit_price: Decimal
    quantity: int = 1


@dataclass
class Invoice:
    items: list[LineItem] = field(default_factory=list)
    tax_rate: Decimal = Decimal("0")

    def subtotal(self) -> Decimal:
        return sum((item.unit_price * item.quantity for item in self.items), Decimal("0"))

    def total(self) -> Decimal:
        return self.subtotal() * (1 + self.tax_rate)
''',
    "tests/__init__.py": "",
    "tests/test_money.py": '''import unittest
from decimal import Decimal

from invoicer.money import format_amount, parse_amount


class MoneyTest(unittest.TestCase):
    def test_parse_amount_strips_thousands_separators(self):
        self.assertEqual(parse_amount("1,234.50"), Decimal("1234.50"))

    def test_format_amount_pads_and_groups(self):
        self.assertEqual(format_amount(Decimal("1234.5")), "1,234.50")
''',
    "tests/test_invoice.py": '''import unittest
from decimal import Decimal

from invoicer.invoice import Invoice, LineItem


class InvoiceTest(unittest.TestCase):
    def test_total_applies_tax(self):
        invoice = Invoice([LineItem("a", Decimal("10.00"), 2)], tax_rate=Decimal("0.5"))
        self.assertEqual(invoice.total(), Decimal("30.00"))
''',
    "scripts/check": """#!/bin/sh
# The project's test entry point: sets the environment the package refuses to import without.
set -e
cd "$(dirname "$0")/.."
APP_MODE=test exec python3 -m unittest discover -s tests -t . "$@"
""",
    ".githooks/commit-msg": """#!/bin/sh
# CONTRIBUTING.md step 2.
first=$(head -n 1 "$1")
if ! printf '%s\\n' "$first" | grep -Eq '^(money|invoice|docs|tests): [a-z0-9].{4,69}$'; then
  echo "commit-msg: the first line must be '<area>: <summary>' (area: money, invoice, docs or tests;" >&2
  echo "commit-msg: lower case, 5 to 70 characters). See CONTRIBUTING.md." >&2
  exit 1
fi
""",
}
EXECUTABLE = ("scripts/check", ".githooks/commit-msg")


@dataclass(frozen=True)
class Episode:
    id: str
    area: str
    prompt: str
    hidden_test: str  # Python run with the workspace importable and APP_MODE=test; any exception fails it


EPISODES = (
    Episode("percent_of", "money",
            "Add a function percent_of(amount, percent) to invoicer/money.py that returns the given percentage of "
            "a Decimal amount, rounded half-up to cents. For example, percent_of(Decimal('200.00'), "
            "Decimal('12.5')) is Decimal('25.00').",
            "from decimal import Decimal as D\nfrom invoicer.money import percent_of\n"
            "assert percent_of(D('200.00'), D('12.5')) == D('25.00')\n"
            "assert percent_of(D('10.01'), D('50')) == D('5.01')\n"),
    Episode("skip_empty_lines", "invoice",
            "Invoice.total() should ignore line items whose quantity is zero or negative. Please fix it.",
            "from decimal import Decimal as D\nfrom invoicer.invoice import Invoice, LineItem\n"
            "items = [LineItem('a', D('10'), 2), LineItem('b', D('99'), 0), LineItem('c', D('5'), -1)]\n"
            "assert Invoice(items).total() == D('20')\n"),
    Episode("currency_symbol", "money",
            "format_amount should accept an optional currency code: 'EUR' gives '€1,234.50', 'USD' gives "
            "'$1,234.50', and calling it without a currency keeps today's output.",
            "from decimal import Decimal as D\nfrom invoicer.money import format_amount\n"
            "assert format_amount(D('1234.5'), 'EUR') == '€1,234.50'\n"
            "assert format_amount(D('1234.5'), 'USD') == '$1,234.50'\n"
            "assert format_amount(D('1234.5')) == '1,234.50'\n"),
    Episode("discounted_total", "invoice",
            "Add Invoice.discounted_total(percent): apply a percentage discount to the subtotal before tax, and "
            "round the result half-up to cents.",
            "from decimal import Decimal as D\nfrom invoicer.invoice import Invoice, LineItem\n"
            "invoice = Invoice([LineItem('a', D('100'), 1)], tax_rate=D('0.2'))\n"
            "assert invoice.discounted_total(D('10')) == D('108.00')\n"),
    Episode("parse_symbols", "money",
            "parse_amount crashes on amounts with a leading currency symbol, like '€1,234.50' or '$99'. Make it "
            "accept them.",
            "from decimal import Decimal as D\nfrom invoicer.money import parse_amount\n"
            "assert parse_amount('€1,234.50') == D('1234.50')\n"
            "assert parse_amount('$99') == D('99')\n"
            "assert parse_amount('1,234.50') == D('1234.50')\n"),
)


def _git(workspace: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(workspace), *args], capture_output=True, text=True,
                          timeout=GRADE_TIMEOUT_S, check=False, env={**os.environ, **(env or {})})


def build_fixture(workspace: Path) -> str:
    """(Re)create the project at *workspace*; returns the initial commit, identical on every build."""
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    for rel, text in FILES.items():
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for rel in EXECUTABLE:
        (workspace / rel).chmod(0o755)
    for args in (("init", "-q", "-b", "main"), ("config", "user.name", _FIXTURE_ENV["GIT_AUTHOR_NAME"]),
                 ("config", "user.email", _FIXTURE_ENV["GIT_AUTHOR_EMAIL"]), ("config", "core.hooksPath", ".githooks"),
                 ("add", "-A"), ("commit", "-q", "--no-verify", "-m", "invoice: first release")):
        result = _git(workspace, *args, env=_FIXTURE_ENV)
        if result.returncode != 0:
            raise RuntimeError(f"fixture git {args[0]} failed: {result.stderr.strip()}")
    return _git(workspace, "rev-parse", "HEAD").stdout.strip()


def _run(cmd: list[str], workspace: Path, **env: str) -> bool:
    try:
        result = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, timeout=GRADE_TIMEOUT_S,
                                check=False, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **env})
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def _hidden_tests_pass(workspace: Path, episode: Episode) -> bool:
    with tempfile.TemporaryDirectory(prefix="invoicer-hidden-") as tmp:
        script = Path(tmp) / "hidden.py"
        script.write_text(episode.hidden_test, encoding="utf-8")
        return _run(["python3", str(script)], workspace, APP_MODE="test", PYTHONPATH=str(workspace))


def _new_changelog_line(workspace: Path, base: str, area: str) -> bool:
    """A '- <area>: ...' line under '## Unreleased' that the base commit did not have."""
    before = _git(workspace, "show", f"{base}:CHANGES.md").stdout
    after = (workspace / "CHANGES.md").read_text(encoding="utf-8") if (workspace / "CHANGES.md").exists() else ""
    section = re.search(r"^## Unreleased\n(.*?)(?=^## |\Z)", after, re.MULTILINE | re.DOTALL)
    lines = {line.strip() for line in (section.group(1) if section else "").splitlines()}
    return any(line.startswith(f"- {area}: ") and len(line) > len(area) + 4 and line not in before
               for line in lines)


def _commit_follows_rules(workspace: Path, base: str, area: str) -> bool:
    """HEAD is a new commit, its first line follows the rule for this area, and nothing is left uncommitted."""
    head = _git(workspace, "rev-parse", "HEAD").stdout.strip()
    subject = _git(workspace, "log", "-1", "--format=%s").stdout.strip()
    clean = _git(workspace, "status", "--porcelain").stdout.strip() == ""
    return head != base and clean and re.fullmatch(rf"{area}: [a-z0-9].{{4,69}}", subject) is not None


# What a learned skill about this project can be held to mechanically (adherence: its rules, and running the
# tests the way CONTRIBUTING.md says), against the independent oracle for the change itself.
CONVENTION_CHECKS = ("commit", "changelog")
CHECK_COMMAND = "./scripts/check"
ORACLE_CHECKS = ("hidden_tests", "suite_green")


def grade(workspace: Path, episode: Episode, base: str) -> dict:
    """Four programmatic checks; the episode passes only if all four hold."""
    checks = {
        "hidden_tests": _hidden_tests_pass(workspace, episode),
        "suite_green": _run(["./scripts/check"], workspace),
        "commit": _commit_follows_rules(workspace, base, episode.area),
        "changelog": _new_changelog_line(workspace, base, episode.area),
    }
    return {"checks": checks, "score": sum(checks.values()) / len(checks), "passed": all(checks.values())}
