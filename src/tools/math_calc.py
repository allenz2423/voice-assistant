import ast
import math
import operator
import re
from typing import Any

# Supported math operators
SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "ceil": math.ceil,
    "floor": math.floor,
}

def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value}")
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPS:
            return SAFE_OPS[op_type](left, right)
        raise ValueError(f"Unsupported operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type in SAFE_OPS:
            return SAFE_OPS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCS:
            args = [_eval_node(arg) for arg in node.args]
            return SAFE_FUNCS[node.func.id](*args)
        raise ValueError("Unsupported function call")
    raise ValueError(f"Unsupported syntax: {type(node)}")

def calculate_math(expression: str) -> str:
    """Evaluates mathematical expressions or unit conversions safely."""
    expr = (expression or "").strip()
    if not expr:
        return "Please provide a mathematical expression or unit conversion."

    # 1. Handle common unit conversion patterns
    # e.g., "50 miles to km", "100 c to f", "4.5 gb to mb"
    m_temp = re.match(r"^([0-9.]+)\s*(c|celsius)\s+(to|in)\s+(f|fahrenheit)$", expr, re.IGNORECASE)
    if m_temp:
        val = float(m_temp.group(1))
        res = (val * 9 / 5) + 32
        return f"{val} degrees Celsius is equal to {res:.1f} degrees Fahrenheit."

    m_temp_rev = re.match(r"^([0-9.]+)\s*(f|fahrenheit)\s+(to|in)\s+(c|celsius)$", expr, re.IGNORECASE)
    if m_temp_rev:
        val = float(m_temp_rev.group(1))
        res = (val - 32) * 5 / 9
        return f"{val} degrees Fahrenheit is equal to {res:.1f} degrees Celsius."

    m_dist = re.match(r"^([0-9.]+)\s*(miles?|mi)\s+(to|in)\s+(km|kilometers?)$", expr, re.IGNORECASE)
    if m_dist:
        val = float(m_dist.group(1))
        return f"{val} miles is equal to {val * 1.60934:.2f} kilometers."

    m_dist_rev = re.match(r"^([0-9.]+)\s*(km|kilometers?)\s+(to|in)\s+(miles?|mi)$", expr, re.IGNORECASE)
    if m_dist_rev:
        val = float(m_dist_rev.group(1))
        return f"{val} kilometers is equal to {val / 1.60934:.2f} miles."

    m_bytes = re.match(r"^([0-9.]+)\s*(gb|gigabytes?)\s+(to|in)\s+(mb|megabytes?)$", expr, re.IGNORECASE)
    if m_bytes:
        val = float(m_bytes.group(1))
        return f"{val} gigabytes is equal to {val * 1024:.0f} megabytes."

    # 2. General safe AST math evaluation
    clean_expr = expr.replace("^", "**").replace("x", "*")
    try:
        parsed = ast.parse(clean_expr, mode="eval")
        result = _eval_node(parsed.body)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"The result of {expr} is {result:,}."
    except Exception as e:
        return f"Could not evaluate expression '{expr}': {e}"
