from app.tools.calculator import evaluate_expression


def test_calculator_supports_basic_math():
    assert evaluate_expression('2 + 3 * 4') == '14'
