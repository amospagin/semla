"""Tests for the lavaan syntax parser."""

import pytest
from semla.syntax import FormulaToken, RHSTerm, parse_syntax


class TestParseSyntax:
    def test_basic_cfa(self):
        model = """
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """
        tokens = parse_syntax(model)
        assert len(tokens) == 3

        assert tokens[0].lhs == "visual"
        assert tokens[0].op == "=~"
        assert [t.var for t in tokens[0].rhs] == ["x1", "x2", "x3"]

        assert tokens[1].lhs == "textual"
        assert tokens[2].lhs == "speed"

    def test_regression(self):
        tokens = parse_syntax("y ~ x1 + x2 + x3")
        assert len(tokens) == 1
        assert tokens[0].op == "~"
        assert tokens[0].lhs == "y"
        assert [t.var for t in tokens[0].rhs] == ["x1", "x2", "x3"]

    def test_covariance(self):
        tokens = parse_syntax("x1 ~~ x2")
        assert len(tokens) == 1
        assert tokens[0].op == "~~"
        assert tokens[0].lhs == "x1"
        assert tokens[0].rhs[0].var == "x2"

    def test_variance(self):
        tokens = parse_syntax("x1 ~~ x1")
        assert tokens[0].lhs == "x1"
        assert tokens[0].rhs[0].var == "x1"

    def test_intercept(self):
        tokens = parse_syntax("y ~1")
        assert len(tokens) == 1
        assert tokens[0].op == "~1"
        assert tokens[0].lhs == "y"
        assert tokens[0].rhs == []

    def test_fixed_loading(self):
        tokens = parse_syntax("f1 =~ 1*x1 + x2 + x3")
        assert tokens[0].rhs[0].var == "x1"
        assert tokens[0].rhs[0].fixed is True
        assert tokens[0].rhs[0].start_value == 1.0
        assert tokens[0].rhs[1].fixed is False

    def test_numeric_modifier(self):
        tokens = parse_syntax("f1 =~ 0.5*x1 + x2")
        assert tokens[0].rhs[0].modifier == 0.5
        assert tokens[0].rhs[0].fixed is True

    def test_label_modifier(self):
        tokens = parse_syntax("f1 =~ a*x1 + b*x2")
        assert tokens[0].rhs[0].modifier == "a"
        assert tokens[0].rhs[0].fixed is False
        assert tokens[0].rhs[1].modifier == "b"

    def test_na_modifier(self):
        tokens = parse_syntax("f1 =~ NA*x1 + x2")
        assert tokens[0].rhs[0].var == "x1"
        assert tokens[0].rhs[0].fixed is False
        assert tokens[0].rhs[0].modifier is None

    def test_comments(self):
        model = """
            # This is a comment
            f1 =~ x1 + x2  # inline comment
        """
        tokens = parse_syntax(model)
        assert len(tokens) == 1
        assert tokens[0].lhs == "f1"

    def test_semicolons(self):
        tokens = parse_syntax("f1 =~ x1 + x2; f2 =~ x3 + x4")
        assert len(tokens) == 2

    def test_blank_lines(self):
        model = """

            f1 =~ x1 + x2

            f2 =~ x3 + x4

        """
        tokens = parse_syntax(model)
        assert len(tokens) == 2

    def test_empty_model(self):
        tokens = parse_syntax("")
        assert len(tokens) == 0

    def test_no_operator_raises(self):
        with pytest.raises(SyntaxError, match="No valid operator"):
            parse_syntax("x1 x2 x3")

    def test_missing_lhs_raises(self):
        with pytest.raises(SyntaxError, match="Missing left-hand side"):
            parse_syntax("=~ x1 + x2")

    def test_missing_rhs_raises(self):
        with pytest.raises(SyntaxError, match="Missing right-hand side"):
            parse_syntax("f1 =~")

    def test_mixed_model(self):
        model = """
            f1 =~ x1 + x2 + x3
            f2 =~ x4 + x5 + x6
            f2 ~ f1
            x1 ~~ x4
        """
        tokens = parse_syntax(model)
        assert len(tokens) == 4
        ops = [t.op for t in tokens]
        assert ops == ["=~", "=~", "~", "~~"]
