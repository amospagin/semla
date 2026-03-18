"""Parser for lavaan-style model syntax.

Supports the following operators:
    =~  latent variable definition (factor loadings)
    ~   regression
    ~~  (co)variance
    ~1  intercept
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RHSTerm:
    """A single term on the right-hand side of a formula."""

    var: str
    modifier: float | str | None = None  # numeric = fixed value, str = label
    fixed: bool = False

    @property
    def start_value(self) -> float | None:
        if self.fixed and isinstance(self.modifier, (int, float)):
            return float(self.modifier)
        return None


@dataclass
class FormulaToken:
    """One parsed formula line: lhs op rhs_terms."""

    lhs: str
    op: str
    rhs: list[RHSTerm] = field(default_factory=list)


# Operators ordered longest-first so =~ matches before ~
_OPERATORS = ["=~", "~~", "~1", "~"]
_OP_PATTERN = re.compile(r"(=~|~~|~1|~)")

# Modifier pattern: optional "number*" or "label*" prefix on a variable
_MODIFIER_RE = re.compile(
    r"^(?:(?P<num>[+-]?\d+(?:\.\d+)?)\*)?(?:(?P<label>[A-Za-z_]\w*)\*)?(?P<var>[A-Za-z_][\w.]*)$"
)

# Numeric-only check
_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def _parse_rhs_term(raw: str) -> RHSTerm:
    """Parse a single RHS term like '0.5*x1', 'a*x1', or 'x1'."""
    raw = raw.strip()

    # Handle "NA*var" -> free parameter (explicit)
    if raw.upper().startswith("NA*"):
        var = raw[3:]
        return RHSTerm(var=var, modifier=None, fixed=False)

    # Try modifier*var pattern
    m = _MODIFIER_RE.match(raw)
    if m:
        var = m.group("var")
        num = m.group("num")
        label = m.group("label")

        if num is not None and label is not None:
            # e.g. "0.5*a*x1" — not supported, treat num as fixed
            return RHSTerm(var=var, modifier=float(num), fixed=True)
        elif num is not None:
            return RHSTerm(var=var, modifier=float(num), fixed=True)
        elif label is not None:
            # Check if label is actually a number (shouldn't happen with regex, but safe)
            if _NUM_RE.match(label):
                return RHSTerm(var=var, modifier=float(label), fixed=True)
            return RHSTerm(var=var, modifier=label, fixed=False)
        else:
            return RHSTerm(var=var, modifier=None, fixed=False)

    # Fallback: treat as plain variable
    return RHSTerm(var=raw, modifier=None, fixed=False)


def parse_syntax(model: str) -> list[FormulaToken]:
    """Parse a lavaan-style model syntax string into a list of FormulaTokens.

    Parameters
    ----------
    model : str
        Model syntax using lavaan conventions. Lines can be separated by
        newlines or semicolons. Comments start with ``#``.

    Returns
    -------
    list[FormulaToken]
        Ordered list of parsed formula tokens.

    Examples
    --------
    >>> tokens = parse_syntax('''
    ...     visual  =~ x1 + x2 + x3
    ...     textual =~ x4 + x5 + x6
    ...     speed   =~ x7 + x8 + x9
    ... ''')
    >>> len(tokens)
    3
    >>> tokens[0].lhs
    'visual'
    >>> tokens[0].op
    '=~'
    >>> [t.var for t in tokens[0].rhs]
    ['x1', 'x2', 'x3']
    """
    tokens: list[FormulaToken] = []

    # Normalize: replace semicolons with newlines
    model = model.replace(";", "\n")

    for raw_line in model.split("\n"):
        # Strip comments
        line = raw_line.split("#")[0].strip()
        if not line:
            continue

        # Find the operator
        m = _OP_PATTERN.search(line)
        if not m:
            raise SyntaxError(
                f"No valid operator found in line: '{raw_line.strip()}'. "
                f"Expected one of: =~ ~ ~~ ~1"
            )

        op = m.group(1)
        lhs = line[: m.start()].strip()
        rhs_str = line[m.end() :].strip()

        if not lhs:
            raise SyntaxError(
                f"Missing left-hand side in: '{raw_line.strip()}'"
            )

        # Handle ~1 (intercept): no RHS needed beyond the "1"
        if op == "~1":
            tokens.append(FormulaToken(lhs=lhs, op="~1", rhs=[]))
            continue

        if not rhs_str:
            raise SyntaxError(
                f"Missing right-hand side in: '{raw_line.strip()}'"
            )

        # Split RHS by +
        rhs_parts = [p.strip() for p in rhs_str.split("+")]
        rhs_terms = [_parse_rhs_term(p) for p in rhs_parts if p]

        tokens.append(FormulaToken(lhs=lhs, op=op, rhs=rhs_terms))

    return tokens
