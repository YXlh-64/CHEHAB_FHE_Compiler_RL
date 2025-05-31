import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

SCALAR_OPS = {"+", "-", "*", "Neg"}
VECTOR_OPS = {"VecAdd", "VecMinus", "VecMul", "VecNeg"}
OTHER_OPS = {"Vec"}
ALL_OPS = SCALAR_OPS.union(VECTOR_OPS).union(OTHER_OPS)
PROTECTED_OPS = VECTOR_OPS.union(OTHER_OPS)

@dataclass
class SubExprInfo:
    op_name: str
    size: int
    depth: int
    contains_zero: bool
    constant_factor: Optional[float]
    children: List[str]
    raw: str

class HierarchicalExpression:
    def __init__(self, expr: str, min_depth: int = 2, ignore_protected: bool = False):
        """
        Initialize with an expression and a minimum depth threshold for abstraction.
        min_depth=2 (default) abstracts nodes at depth 2 or deeper,
        while min_depth=3 leaves one additional level unabstracted.
        
        If ignore_protected=True, then even nodes whose operators are in PROTECTED_OPS 
        will be abstracted (if they meet the min_depth criterion). This is useful when
        re-processing a subexpression for further optimization.
        """
        self.original_expr = expr
        self.min_depth = min_depth
        self.ignore_protected = ignore_protected
        self.current_ast = self._parse(expr)
        self._ph_map: Dict[str, List] = {}    # Maps placeholder id to the processed node.
        self.placeholders: Dict[str, SubExprInfo] = {}  # Analysis info including raw expansion.
        self._expr_hash_map: Dict[str, str] = {}  # Maps hash to placeholder id.
        self.current_ast = self._abstract(self.current_ast, depth=0, parent_op=None)

    def _parse(self, expr: str) -> List:
        tokens = re.split(r"([()]|\s+)", expr)
        tokens = [t.strip() for t in tokens if t.strip()]
        stack = []
        current = []
        for t in tokens:
            if t == "(":
                stack.append(current)
                current = []
            elif t == ")":
                if stack:
                    last = stack.pop()
                    last.append(current)
                    current = last
            else:
                current.append(t)
        return current[0] if current else []

    def _abstract(self, node: Union[List, str], depth: int, parent_op: Optional[str]) -> Union[List, str]:
        if isinstance(node, str):
            return node
        op = node[0]
        processed_children = [
            self._abstract(child, depth + 1, op) if isinstance(child, list) else child
            for child in node[1:]
        ]
        new_node = [op] + processed_children

        # By default, if not ignoring protected, do not abstract nodes whose operator 
        # (or parent's operator) is in PROTECTED_OPS.
        if not self.ignore_protected and (op in PROTECTED_OPS or (parent_op is not None and parent_op in PROTECTED_OPS)):
            return new_node

        original_str = self._unparse(new_node)
        if depth >= self.min_depth and len(new_node) > 2:
            hash_digest = hashlib.md5(original_str.encode()).hexdigest()[:6]
            if hash_digest in self._expr_hash_map:
                ph_id = self._expr_hash_map[hash_digest]
            else:
                ph_id = f"F_{hash_digest}"
                self._expr_hash_map[hash_digest] = ph_id
                self._ph_map[ph_id] = new_node
                self._analyze_placeholder(ph_id, new_node, depth)
            return ph_id
        else:
            return new_node

    def _analyze_placeholder(self, ph_id: str, node: List, depth: int):
        op_name = node[0]
        size = self._calc_size(node)
        contains_zero = self._contains_zero(node)
        const_factor = self._find_constant_factor(node)
        child_phs = [c for c in node[1:] if isinstance(c, str) and c.startswith("F_")]
        self.placeholders[ph_id] = SubExprInfo(
            op_name=op_name,
            size=size,
            depth=depth,
            contains_zero=contains_zero,
            constant_factor=const_factor,
            children=child_phs,
            raw=self._unparse(node)
        )

    def _calc_size(self, node: Union[List, str]) -> int:
        if isinstance(node, str):
            return 1
        return 1 + sum(self._calc_size(child) for child in node[1:])

    def _contains_zero(self, node: Union[List, str]) -> bool:
        if isinstance(node, str):
            return node == "0"
        return any(self._contains_zero(child) for child in node[1:])

    def _find_constant_factor(self, node: Union[List, str]) -> Optional[float]:
        if isinstance(node, str) or len(node) < 2:
            return None
        op = node[0]
        numerics = []
        for child in node[1:]:
            if isinstance(child, str):
                try:
                    numerics.append(float(child))
                except ValueError:
                    pass
        if op == "+" and numerics:
            return sum(numerics)
        if op == "*" and len(numerics) >= 2:
            prod = 1.0
            for val in numerics:
                prod *= val
            return prod
        return None

    def _unparse(self, node: Union[List, str]) -> str:
        if isinstance(node, str):
            return node
        return f"({' '.join(self._unparse(child) for child in node)})"

    def get_abstracted(self) -> str:
        return self._unparse(self.current_ast)

    def get_context(self) -> Dict[str, Dict]:
        used = self._find_placeholders_in_ast(self.current_ast, set())
        out = {}
        for ph in used:
            info = self.placeholders[ph]
            out[ph] = {
                "op_name": info.op_name,
                "size": info.size,
                "contains_zero": info.contains_zero,
                "constant_factor": info.constant_factor,
                "children": info.children,
            }
        return out

    def _find_placeholders_in_ast(self, node: Union[List, str], acc: set) -> set:
        if isinstance(node, str):
            if node.startswith("F_"):
                acc.add(node)
            return acc
        for child in node[1:]:
            self._find_placeholders_in_ast(child, acc)
        return acc

    def _replace_in_ast(self, node: Union[List, str], target: str, replacement: Union[List, str]) -> bool:
        replaced = False
        if isinstance(node, str):
            return False
        for i in range(len(node)):
            if node[i] == target:
                node[i] = replacement
                replaced = True
            elif isinstance(node[i], list):
                if self._replace_in_ast(node[i], target, replacement):
                    replaced = True
        return replaced
    def expand_expr_with_placeholders(self, expr_with_placeholders: str) -> str:
        """
        Given an expression (string) that contains placeholders
        (e.g. F_abc123) plus variables, expand all placeholders
        using self.placeholders, returning a fully expanded expression.
        """

        def _expand_node(node):
            """
            Recursively expand placeholders in the parsed AST `node`.
            If node is a placeholder and is in self.placeholders,
            replace it with the parsed-and-expanded form of its raw expression.
            """
            if isinstance(node, str):
                # If it is a placeholder we know about, expand it:
                if node.startswith("F_") and node in self.placeholders:
                    # Parse the placeholder's raw expression, then recursively expand that AST
                    sub_ast = self._parse(self.placeholders[node].raw)
                    return _expand_node(sub_ast)
                else:
                    # Regular variable or numeric literal
                    return node
            else:
                # It's a list => [op, child1, child2, ...]
                new_list = []
                for child in node:
                    new_list.append(_expand_node(child))
                return new_list

        # 1) Parse the incoming expression into an AST
        ast = self._parse(expr_with_placeholders)

        # 2) Recursively expand all placeholders
        fully_expanded_ast = _expand_node(ast)

        # 3) Convert back to string
        return self._unparse(fully_expanded_ast)

    def expand_one_level(self, ph_id: str, new_min_depth: int) -> str:
        """
        Expand the placeholder ph_id by re-processing its raw expansion using a new
        minimum depth (new_min_depth). This lets you gradually reveal deeper levels.
        The expansion will obey the new_min_depth setting.
        """
        if ph_id not in self.placeholders:
            return self.get_abstracted()
        raw = self.placeholders[ph_id].raw
        # For sub-expressions, set ignore_protected=True so that inner vector ops are abstracted as needed.
        temp_he = HierarchicalExpression(raw, min_depth=new_min_depth, ignore_protected=True)
        self._replace_in_ast(self.current_ast, ph_id, temp_he.current_ast)
        #self.placeholders.pop(ph_id, None)
        #for h, pid in list(self._expr_hash_map.items()):
        #    if pid == ph_id:
        #        self._expr_hash_map.pop(h)
        return self.get_abstracted()

    def expand_placeholder_in_subvec(self, sub_vec_expr: str, placeholder: str) -> str:
        if placeholder not in self.placeholders:
            return sub_vec_expr
        expansion = self.placeholders[placeholder].raw
        pattern = r'\b' + re.escape(placeholder) + r'\b'
        new_expr = re.sub(pattern, expansion, sub_vec_expr)
        return new_expr

    def collect_all_vector_subexpr_strings(self, node: Optional[Union[List, str]] = None) -> List[str]:
        if node is None:
            node = self.current_ast
        results = []
        if isinstance(node, str):
            if node.startswith("F_") and node in self.placeholders and \
               self.placeholders[node].op_name in (OTHER_OPS.union(VECTOR_OPS)):
                results.append(self.placeholders[node].raw)
            return results
        if len(node) > 0 and node[0] in (OTHER_OPS.union(VECTOR_OPS)):
            results.append(self._unparse(node))
        for child in node[1:]:
            if isinstance(child, list) or (isinstance(child, str) and child.startswith("F_")):
                results.extend(self.collect_all_vector_subexpr_strings(child))
        return results

    def get_expandable_placeholders_in_subvec(self, sub_vec_expr: str, max_depth: int=3) -> List[str]:
        def _collect_placeholders(node, current_depth):
            placeholders = []
            if isinstance(node, str):
                if node.startswith("F_") and current_depth < max_depth - 1:
                    placeholders.append(node)
                return placeholders
            for child in node:
                if isinstance(child, str):
                    if child.startswith("F_") and current_depth < max_depth - 1:
                        placeholders.append(child)
                        if child in self.placeholders:
                            child_ast = self._parse(self.placeholders[child].raw)
                            placeholders.extend(_collect_placeholders(child_ast, current_depth + 1))
                elif isinstance(child, list):
                    placeholders.extend(_collect_placeholders(child, current_depth + 1))
            return placeholders

        ast = self._parse(sub_vec_expr)
        return _collect_placeholders(ast, 0)
    def collect_vector_placeholders_topdown(self) -> List[tuple]:
        """
        Return a list of (ph_id, SubExprInfo) for placeholders whose op_name is
        in PROTECTED_OPS. We traverse top-down: if we encounter a vector placeholder,
        we include it, then recursively traverse into its raw expression to find
        nested placeholders inside.
        """
        results = []

        def visit(node):
            # If node is a placeholder string like "F_abc123"
            if isinstance(node, str) and node.startswith("F_"):
                ph_id = node
                # Check if it's a known placeholder
                if ph_id in self.placeholders:
                    info = self.placeholders[ph_id]
                    # If this placeholder is a vector-level subexpression
                    if info.op_name in PROTECTED_OPS:
                        # Collect it
                        results.append((ph_id, info))
                    # Regardless of whether we appended it, we might still want
                    # to look inside the raw expression for nested placeholders.
                    sub_ast = self._parse(info.raw)
                    # Recurse into the raw AST for deeper placeholders
                    visit(sub_ast)

            # If node is a list => [operator, child1, child2, ...]
            elif isinstance(node, list):
                for child in node[1:]:
                    visit(child)

        # Start by visiting the top-level current_ast
        visit(self.current_ast)
        return results




# --- Main code demonstrating staged expansion for sub Vec expressions ---
if __name__ == "__main__":
    # Example expression (you can substitute your own).
    expr = "(VecMul (VecAdd (VecAdd (VecMul (VecAdd (Vec (* F_b6b52e 1) 1 (* F_6cdcdb F_daba82) (+ 0 F_a7c6d4) (+ F_d617bf F_8c14ca) 0 (+ F_a7c6d4 F_edfe61) (+ F_d617bf F_8c14ca) (+ F_a73f46 F_f7623e)) (VecAdd (Vec 0 0 (* F_31f4ac F_0ea435) 0 0 0 0 0 0) (Vec 0 0 0 0 0 0 0 0 0))) (Vec 1 1 1 1 1 1 1 1 1)) (VecMul (Vec 0 0 0 (* F_1686e6 F_83bd67) 0 0 0 0 1) (Vec 1 1 1 1 1 1 1 1 0))) (VecAdd (VecAdd(Vec 0 0 0 0 0 (+ F_a73f46 F_f7623e) 0 0 0) (Vec 0 0 0 0 0 0 0 0 0)) (Vec 0 0 0 0 0 0 0 0 0))) (Vec 1 (+ F_d617bf F_8c14ca) 1 1 1 1 1 1 1))"
    
    # Start with a higher min_depth (e.g., 3) so that the expression is highly collapsed.
    he = HierarchicalExpression(expr, min_depth=3)
    
    print("Initial Abstracted Expression:")
    print(he.get_abstracted())
    subvec_placeholders = he.collect_vector_placeholders_topdown()
    for (ph, info) in subvec_placeholders:
        print(f"Found vector placeholder {ph} of op {info.op_name}")
        print(f"   raw: {info.raw}")
        print(f"   size={info.size}, depth={info.depth}, children={info.children}")
    #expressions = he.collect_all_vector_subexpr_strings()
    #print(expressions)
    # Now, suppose you want to gradually expand each placeholder one level.
    # For demonstration, we repeatedly list the placeholders in the vector subexpression,
    # then expand them one by one using a lower min_depth (e.g., 2) for the sub-expression.
    #current = he.get_abstracted()
    #print(he.expand_expr_with_placeholders("(VecMul (VecMul (VecAdd F_18619c F_6668bb) (VecAdd F_6eb9a9 F_fcd6e0)) (VecAdd (VecAdd F_5ccd56 F_b19fa8) (VecMul F_953dc7 F_5363b7)))"))
    #print("expressions",expressions)
   # placeholders = he.get_expandable_placeholders_in_subvec(expressions[0])
    #print(placeholders)
    #for ph in placeholders:
        #print(f"\nExpanding placeholder {ph} one level (new_min_depth=2):")
        #he.expand_one_level(ph, new_min_depth=2)
        #print(he.get_abstracted())
        #placeholders.remove(ph)
    #expressions = he.collect_all_vector_subexpr_strings()
    #print(expressions[0])
    
    print("\nFully Extended Expression:")
    print(he.get_abstracted())
   
