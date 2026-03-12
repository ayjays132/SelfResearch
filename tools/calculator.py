import ast
import operator
from tools.base_tool import BaseTool

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluates basic mathematical expressions. Useful for checking data stats, formulas, or doing math."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g., '2 + 2 * 5'"
            }
        },
        "required": ["expression"]
    }

    def execute(self, expression: str, **kwargs) -> str:
        try:
            # Safe evaluation of math expressions
            allowed_operators = {
                ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                ast.Div: operator.truediv, ast.Pow: operator.pow, ast.BitXor: operator.xor,
                ast.USub: operator.neg
            }
            
            def eval_expr(node):
                if isinstance(node, ast.Constant): # <number> (Python 3.8+)
                    return node.value
                elif isinstance(node, ast.BinOp): # <left> <operator> <right>
                    return allowed_operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
                    return allowed_operators[type(node.op)](eval_expr(node.operand))
                else:
                    raise TypeError(node)
                    
            node = ast.parse(expression, mode='eval').body
            result = eval_expr(node)
            return f"Result: {result}"
        except Exception as e:
            return f"Error evaluating expression: {e}"
