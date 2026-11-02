"""Static Python code features; submitted source is never executed."""
import ast
import io
import keyword
import re
import tokenize

STRUCTURAL_FEATURES = [
    "comment_to_code_ratio", "average_line_length", "average_identifier_length",
    "snake_case_ratio", "camel_case_ratio", "function_count", "loop_count",
    "conditional_count", "class_count", "exception_count", "ast_node_count",
    "max_nesting_depth", "code_length", "line_count", "token_count",
    "unique_token_ratio", "blank_line_ratio", "docstring_count",
]


def extract_features(source: str) -> dict:
    lines = source.splitlines()
    nonblank = [line for line in lines if line.strip()]
    code_lines = sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in lines)
    identifiers, tokens = [], []
    comments = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                comments += 1
            elif tok.type == tokenize.NAME and not keyword.iskeyword(tok.string):
                identifiers.append(tok.string)
            if tok.type not in (tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
                tokens.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        pass
    unique = set(tokens)
    id_len = sum(map(len, identifiers)) / max(1, len(identifiers))
    snake = sum(bool(re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", x)) for x in identifiers)
    camel = sum(bool(re.fullmatch(r"[a-z]+(?:[A-Z][a-z0-9]*)+", x)) for x in identifiers)
    counts = {"function_count": 0, "loop_count": 0, "conditional_count": 0, "class_count": 0, "exception_count": 0, "ast_node_count": 0, "max_nesting_depth": 0, "docstring_count": 0}
    try:
        root = ast.parse(source)
        counts["ast_node_count"] = sum(1 for _ in ast.walk(root))
        counts["function_count"] = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(root))
        counts["loop_count"] = sum(isinstance(n, (ast.For, ast.AsyncFor, ast.While)) for n in ast.walk(root))
        counts["conditional_count"] = sum(isinstance(n, (ast.If, ast.IfExp, ast.Match)) for n in ast.walk(root))
        counts["class_count"] = sum(isinstance(n, ast.ClassDef) for n in ast.walk(root))
        counts["exception_count"] = sum(isinstance(n, (ast.Try, ast.Raise, ast.ExceptHandler)) for n in ast.walk(root))
        counts["docstring_count"] = sum(ast.get_docstring(n) is not None for n in ast.walk(root) if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        def depth(node, level=0):
            here = level + isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef))
            return max([here] + [depth(c, here) for c in ast.iter_child_nodes(node)])
        counts["max_nesting_depth"] = depth(root)
    except SyntaxError:
        pass
    return {
        "comment_to_code_ratio": comments / max(1, code_lines),
        "average_line_length": sum(map(len, nonblank)) / max(1, len(nonblank)),
        "average_identifier_length": id_len,
        "snake_case_ratio": snake / max(1, len(identifiers)), "camel_case_ratio": camel / max(1, len(identifiers)),
        **counts, "code_length": len(source), "line_count": len(lines), "token_count": len(tokens),
        "unique_token_ratio": len(unique) / max(1, len(tokens)),
        "blank_line_ratio": (len(lines) - len(nonblank)) / max(1, len(lines)),
    }
