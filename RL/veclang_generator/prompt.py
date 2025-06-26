from langchain_core.prompts import ChatPromptTemplate







def build_veclang_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "You are a rigorous validator for VecLang expressions. First ANALYZE then GENERATE. **Enforce structural uniqueness beyond variable renaming.**"),
        ("human", PROMPT),
    ])


PROMPT = """**VecLang Specification- Strict Validation Required**
1. **Core Structure**:
   - MUST start with `(Vec` followed by EXACTLY {vec_size} sub-expressions  
   - Example: `(Vec expr1 expr2 … expr{vec_size})`  
   - **Minimum depth of 6**, at least 3 expressions must have depth > 12

2. **Validation Checklist** (MUST satisfy ALL):
   ✅ **{vec_size} Elements**: Count elements after Vec before closing parenthesis  
   ✅ **Parentheses Balance**: Equal `(` and `)`, no nested Vec  
   ✅ **Operators**: Only `+`, `-`, `*` (no division)  
   ✅ **Variables**: Use patterns like `in_0_0`, `c1_0`, `v2_3`  
   ✅ **Nesting Depth**: 6 ≤ depth ≤ 30 for all sub-expressions  
   ✅ **Uniqueness**: No match with history or examples  
   ✅ **No Constants-Only**: At least 3 variables per expression  
   ✅ **Structural Uniqueness**: No isomorphic match with history/examples when normalized  
   ✅ **Depth Spectrum**: At least 2 expressions > 15 depth  
   ✅ **Operation Asymmetry**: No identical operator patterns when normalized

3. **Syntax Enforcement**:
vec_expr = "(Vec", expr₁, expr₂, …, expr{vec_size}, ")"
   vec_expr = "(Vec", expression, expression, expression, 
              expression, expression, expression,
              expression, expression, expression, ")";
   expression = constant | variable | operation;
   operation = "(", ("+" | "-" | "*"), expression, expression, ")";
   All operations have only 2 children no more
3. **Depth Enforcement Examples**:
   - Valid: `(Vec (+ (* (+ a (- b c)) d) (...)) ...)` [depth 9]
   - Invalid: `(Vec (+ a b) (- c d) ...)` [depth 2] 
   - **Required**: expressions be with nested operations like `(* (+ (- (...) ...) (* ...)) ...)`
4. **Quality Control Steps (REQUIRED PROCESS)**:

  - Generate candidate expression

  - Verify element count (exactly {vec_size})

  - Validate parentheses balance

  - Check operator validity

  - Measure nesting depth

  - Cross-reference with history

  - Ensure structural uniqueness
  - All operations have only 2 children no more

  - Final syntax check
  - Convert expression to canonical form (replace variables with V, constants with C) 
  - Verify canonical form doesn't match any previous entries
  - Generate candidate
   - **Convert to canonical skeleton**: 
     `(Vec (OP (OP V C) ...) ...)` → `(Vec (OP (OP V1 C1) ...) ...)`
   - Check against canonical pattern database
   
5. **Common Failure Modes to Avoid**:
  ❌ Vec with <{vec_size} elements ❌ Unbalanced parentheses
  ❌ Division operator ❌ Duplicate structures
  ❌ Invalid variables ❌ Shallow nesting (depth <4)
  ❌ Constant-only terms ❌ Trivial example variations
  ❌ Structurally identical expressions with variable/constant substitution
  ❌ **Shallow Structure**: `(Vec (+ a b) (- c d) ...)` (depth 2)
  ❌ **Depth Inflation**: Fake depth via `(+ (+ (+ (...)))` without meaningful operations
  ❌ **Mirror Structures**: Same operator tree with different variable paths
6. **Examples of INVALID Expressions**:

  - (Vec (+ a b (* c d) → Missing closing )

  - (Vec 1 2 3 4 5 6 7 8 9) → No variables

  - (Vec (+ a b) c d e f g h i) → Depth 1 (too shallow)
  
  - (Vec 0 0 0 0 0 0 0 0 (- 9 4)) -> no variables
7. **Generation Requirements**:
   - **expressions must contain chained operations** like: 
     `(* (- (+ (...)) (...))`
   - **At least 1 deep expression per**:
     - Nested multiplication contexts
     - Mixed operator hierarchies
     - Alternated constant/variable positioning
## Rewriting Rules (for context only)
Rewrite {{ name: "add-0-0+0", searcher: 0, applier: (+ 0 0) }}
Rewrite {{ name: "add-a-a+0", searcher: ?a, applier: (+ ?a 0) }}
Rewrite {{ name: "add-a*b-0+a*b", searcher: (* ?a ?b), applier: (+ 0 (* ?a ?b)) }}
Rewrite {{ name: "add-a-b-0+a-b", searcher: (- ?a ?b), applier: (+ 0 (- ?a ?b)) }}
Rewrite {{ name: "add--a-0+-a", searcher: (- ?a), applier: (+ 0 (- ?a)) }}
Rewrite {{ name: "neg-0-0+0", searcher: 0, applier: (- 0) }}
Rewrite {{ name: "sub-0-0-0", searcher: 0, applier: (- 0 0) }}
Rewrite {{ name: "sub-a-a-0", searcher: ?a, applier: (- ?a 0) }}
Rewrite {{ name: "sub-a*b-0-a*b", searcher: (* ?a ?b), applier: (- 0 (* ?a ?b)) }}
Rewrite {{ name: "sub-a+b-0-a+b", searcher: (+ ?a ?b), applier: (- 0 (+ ?a ?b)) }}
Rewrite {{ name: "sub--a-0--a", searcher: (- ?a), applier: (- 0 (- ?a)) }}
Rewrite {{ name: "add-vectorize-1", searcher: (Vec (+ ?a0 ?b0)), applier: (VecAdd (Vec ?a0) (Vec ?b0)) }}
Rewrite {{ name: "add-vectorize-2", searcher: (Vec (+ ?a0 ?b0) (+ ?a1 ?b1)), applier: (VecAdd (Vec ?a0 ?a1) (Vec ?b0 ?b1)) }}
Rewrite {{ name: "add-vectorize-4", searcher: (Vec (+ ?a0 ?b0) (+ ?a1 ?b1) (+ ?a2 ?b2) (+ ?a3 ?b3)), applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3) (Vec ?b0 ?b1 ?b2 ?b3)) }}
Rewrite {{ name: "add-vectorize-8", searcher: (Vec (+ ?a0 ?b0) (+ ?a1 ?b1) (+ ?a2 ?b2) (+ ?a3 ?b3) (+ ?a4 ?b4) (+ ?a5 ?b5) (+ ?a6 ?b6) (+ ?a7 ?b7)), applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7) (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7)) }}
Rewrite {{ name: "add-vectorize-16", searcher: (Vec (+ ?a0 ?b0) (+ ?a1 ?b1) (+ ?a2 ?b2) (+ ?a3 ?b3) (+ ?a4 ?b4) (+ ?a5 ?b5) (+ ?a6 ?b6) (+ ?a7 ?b7) (+ ?a8 ?b8) (+ ?a9 ?b9) (+ ?a10 ?b10) (+ ?a11 ?b11) (+ ?a12 ?b12) (+ ?a13 ?b13) (+ ?a14 ?b14) (+ ?a15 ?b15)), applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8 ?a9 ?a10 ?a11 ?a12 ?a13 ?a14 ?a15) (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15)) }}
Rewrite {{ name: "add-vectorize-32", searcher: (Vec (+ ?a0 ?b0) (+ ?a1 ?b1) (+ ?a2 ?b2) (+ ?a3 ?b3) (+ ?a4 ?b4) (+ ?a5 ?b5) (+ ?a6 ?b6) (+ ?a7 ?b7) (+ ?a8 ?b8) (+ ?a9 ?b9) (+ ?a10 ?b10) (+ ?a11 ?b11) (+ ?a12 ?b12) (+ ?a13 ?b13) (+ ?a14 ?b14) (+ ?a15 ?b15) (+ ?a16 ?b16) (+ ?a17 ?b17) (+ ?a18 ?b18) (+ ?a19 ?b19) (+ ?a20 ?b20) (+ ?a21 ?b21) (+ ?a22 ?b22) (+ ?a23 ?b23) (+ ?a24 ?b24) (+ ?a25 ?b25) (+ ?a26 ?b26) (+ ?a27 ?b27) (+ ?a28 ?b28) (+ ?a29 ?b29) (+ ?a30 ?b30) (+ ?a31 ?b31)), applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8 ?a9 ?a10 ?a11 ?a12 ?a13 ?a14 ?a15 ?a16 ?a17 ?a18 ?a19 ?a20 ?a21 ?a22 ?a23 ?a24 ?a25 ?a26 ?a27 ?a28 ?a29 ?a30 ?a31) (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15 ?b16 ?b17 ?b18 ?b19 ?b20 ?b21 ?b22 ?b23 ?b24 ?b25 ?b26 ?b27 ?b28 ?b29 ?b30 ?b31)) }}
Rewrite {{ name: "rot-add-vectorize-1", searcher: (Vec (+ ?a0 ?b0)), applier: (VecAdd (Vec ?a0 ?b0) (<< (Vec ?a0 ?b0) 1)) }}
Rewrite {{ name: "rot-add-vectorize-2", searcher: (Vec (+ ?a0 ?b0) (+ ?a1 ?b1)), applier: (VecAdd (Vec ?a0 ?a1 ?b0 ?b1) (<< (Vec ?a0 ?a1 ?b0 ?b1) 2)) }}
Rewrite {{ name: "rot-add-vectorize-4", searcher: (Vec (+ ?a0 ?b0) (+ ?a1 ?b1) (+ ?a2 ?b2) (+ ?a3 ?b3)), applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?b0 ?b1 ?b2 ?b3) (<< (Vec ?a0 ?a1 ?a2 ?a3 ?b0 ?b1 ?b2 ?b3) 4)) }}
Rewrite {{ name: "rot-add-vectorize-8", searcher: (Vec (+ ?a0 ?b0) (+ ?a1 ?b1) (+ ?a2 ?b2) (+ ?a3 ?b3) (+ ?a4 ?b4) (+ ?a5 ?b5) (+ ?a6 ?b6) (+ ?a7 ?b7)), applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7) (<< (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7) 8)) }}
Rewrite {{ name: "rot-add-vectorize-16", searcher: (Vec (+ ?a0 ?b0) (+ ?a1 ?b1) (+ ?a2 ?b2) (+ ?a3 ?b3) (+ ?a4 ?b4) (+ ?a5 ?b5) (+ ?a6 ?b6) (+ ?a7 ?b7) (+ ?a8 ?b8) (+ ?a9 ?b9) (+ ?a10 ?b10) (+ ?a11 ?b11) (+ ?a12 ?b12) (+ ?a13 ?b13) (+ ?a14 ?b14) (+ ?a15 ?b15)), applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8 ?a9 ?a10 ?a11 ?a12 ?a13 ?a14 ?a15 ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15) (<< (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8 ?a9 ?a10 ?a11 ?a12 ?a13 ?a14 ?a15 ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15) 16)) }}
Rewrite {{ name: "sub-vectorize-1", searcher: (Vec (- ?a0 ?b0)), applier: (VecMinus (Vec ?a0) (Vec ?b0)) }}
Rewrite {{ name: "sub-vectorize-2", searcher: (Vec (- ?a0 ?b0) (- ?a1 ?b1)), applier: (VecMinus (Vec ?a0 ?a1) (Vec ?b0 ?b1)) }}
Rewrite {{ name: "sub-vectorize-4", searcher: (Vec (- ?a0 ?b0) (- ?a1 ?b1) (- ?a2 ?b2) (- ?a3 ?b3)), applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3) (Vec ?b0 ?b1 ?b2 ?b3)) }}
Rewrite {{ name: "sub-vectorize-8", searcher: (Vec (- ?a0 ?b0) (- ?a1 ?b1) (- ?a2 ?b2) (- ?a3 ?b3) (- ?a4 ?b4) (- ?a5 ?b5) (- ?a6 ?b6) (- ?a7 ?b7)), applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7) (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7)) }}
Rewrite {{ name: "sub-vectorize-16", searcher: (Vec (- ?a0 ?b0) (- ?a1 ?b1) (- ?a2 ?b2) (- ?a3 ?b3) (- ?a4 ?b4) (- ?a5 ?b5) (- ?a6 ?b6) (- ?a7 ?b7) (- ?a8 ?b8) (- ?a9 ?b9) (- ?a10 ?b10) (- ?a11 ?b11) (- ?a12 ?b12) (- ?a13 ?b13) (- ?a14 ?b14) (- ?a15 ?b15)), applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8 ?a9 ?a10 ?a11 ?a12 ?a13 ?a14 ?a15) (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15)) }}
Rewrite {{ name: "sub-vectorize-32", searcher: (Vec (- ?a0 ?b0) (- ?a1 ?b1) (- ?a2 ?b2) (- ?a3 ?b3) (- ?a4 ?b4) (- ?a5 ?b5) (- ?a6 ?b6) (- ?a7 ?b7) (- ?a8 ?b8) (- ?a9 ?b9) (- ?a10 ?b10) (- ?a11 ?b11) (- ?a12 ?b12) (- ?a13 ?b13) (- ?a14 ?b14) (- ?a15 ?b15) (- ?a16 ?b16) (- ?a17 ?b17) (- ?a18 ?b18) (- ?a19 ?b19) (- ?a20 ?b20) (- ?a21 ?b21) (- ?a22 ?b22) (- ?a23 ?b23) (- ?a24 ?b24) (- ?a25 ?b25) (- ?a26 ?b26) (- ?a27 ?b27) (- ?a28 ?b28) (- ?a29 ?b29) (- ?a30 ?b30) (- ?a31 ?b31)), applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8 ?a9 ?a10 ?a11 ?a12 ?a13 ?a14 ?a15 ?a16 ?a17 ?a18 ?a19 ?a20 ?a21 ?a22 ?a23 ?a24 ?a25 ?a26 ?a27 ?a28 ?a29 ?a30 ?a31) (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15 ?b16 ?b17 ?b18 ?b19 ?b20 ?b21 ?b22 ?b23 ?b24 ?b25 ?b26 ?b27 ?b28 ?b29 ?b30 ?b31)) }}
Rewrite {{ name: "rot-min-vectorize-1", searcher: (Vec (- ?a0 ?b0)), applier: (VecMinus (Vec ?a0 ?b0) (<< (Vec ?a0 ?b0) 1)) }}
Rewrite {{ name: "rot-min-vectorize-2", searcher: (Vec (- ?a0 ?b0) (- ?a1 ?b1)), applier: (VecMinus (Vec ?a0 ?a1 ?b0 ?b1) (<< (Vec ?a0 ?a1 ?b0 ?b1) 2)) }}
Rewrite {{ name: "rot-min-vectorize-4", searcher: (Vec (- ?a0 ?b0) (- ?a1 ?b1) (- ?a2 ?b2) (- ?a3 ?b3)), applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?b0 ?b1 ?b2 ?b3) (<< (Vec ?a0 ?a1 ?a2 ?a3 ?b0 ?b1 ?b2 ?b3) 4)) }}
Rewrite {{ name: "rot-min-vectorize-8", searcher: (Vec (- ?a0 ?b0) (- ?a1 ?b1) (- ?a2 ?b2) (- ?a3 ?b3) (- ?a4 ?b4) (- ?a5 ?b5) (- ?a6 ?b6) (- ?a7 ?b7)), applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7) (<< (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7) 8)) }}
Rewrite {{ name: "rot-min-vectorize-16", searcher: (Vec (- ?a0 ?b0) (- ?a1 ?b1) (- ?a2 ?b2) (- ?a3 ?b3) (- ?a4 ?b4) (- ?a5 ?b5) (- ?a6 ?b6) (- ?a7 ?b7) (- ?a8 ?b8) (- ?a9 ?b9) (- ?a10 ?b10) (- ?a11 ?b11) (- ?a12 ?b12) (- ?a13 ?b13) (- ?a14 ?b14) (- ?a15 ?b15)), applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8 ?a9 ?a10 ?a11 ?a12 ?a13 ?a14 ?a15 ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15) (<< (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8 ?a9 ?a10 ?a11 ?a12 ?a13 ?a14 ?a15 ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15) 16)) }}
Rewrite {{ name: "mul-0-0*0", searcher: 0, applier: (* 0 0) }}
Rewrite {{ name: "mul-a-a*1", searcher: ?a, applier: (* ?a 1) }}
Rewrite {{ name: "mul-a+b-1-a+b", searcher: (+ ?a ?b), applier: (* 1 (+ ?a ?b)) }}
Rewrite {{ name: "mul-a-b-1-a-b", searcher: (- ?a ?b), applier: (* 1 (- ?a ?b)) }}
Rewrite {{ name: "mul-vectorize-1", searcher: (Vec (* ?a0 ?b0)), applier: (VecMul (Vec ?a0) (Vec ?b0)) }}
Rewrite {{ name: "mul-vectorize-2", searcher: (Vec (* ?a0 ?b0) (* ?a1 ?b1)), applier: (VecMul (Vec ?a0 ?a1) (Vec ?b0 ?b1)) }}
Rewrite {{ name: "mul-vectorize-4", searcher: (Vec (* ?a0 ?b0) (* ?a1 ?b1) (* ?a2 ?b2) (* ?a3 ?b3)), applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3) (Vec ?b0 ?b1 ?b2 ?b3)) }}
Rewrite {{ name: "mul-vectorize-8", searcher: (Vec (* ?a0 ?b0) (* ?a1 ?b1) (* ?a2 ?b2) (* ?a3 ?b3) (* ?a4 ?b4) (* ?a5 ?b5) (* ?a6 ?b6) (* ?a7 ?b7)), applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7) (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7)) }}
Rewrite {{ name: "mul-vectorize-16", searcher: (Vec (* ?a0 ?b0) (* ?a1 ?b1) (* ?a2 ?b2) (* ?a3 ?b3) (* ?a4 ?b4) (* ?a5 ?b5) (* ?a6 ?b6) (* ?a7 ?b7) (* ?a8 ?b8) (* ?a9 ?b9) (* ?a10 ?b10) (* ?a11 ?b11) (* ?a12 ?b12) (* ?a13 ?b13) (* ?a14 ?b14) (* ?a15 ?b15)), applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8 ?a9 ?a10 ?a11 ?a12 ?a13 ?a14 ?a15) (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15)) }}
Rewrite {{ name: "mul-vectorize-32", searcher: (Vec (* ?a0 ?b0) (* ?a1 ?b1) (* ?a2 ?b2) (* ?a3 ?b3) (* ?a4 ?b4) (* ?a5 ?b5) (* ?a6 ?b6) (* ?a7 ?b7) (* ?a8 ?b8) (* ?a9 ?b9) (* ?a10 ?b10) (* ?a11 ?b11) (* ?a12 ?b12) (* ?a13 ?b13) (* ?a14 ?b14) (* ?a15 ?b15) (* ?a16 ?b16) (* ?a17 ?b17) (* ?a18 ?b18) (* ?a19 ?b19) (* ?a20 ?b20) (* ?a21 ?b21) (* ?a22 ?b22) (* ?a23 ?b23) (* ?a24 ?b24) (* ?a25 ?b25) (* ?a26 ?b26) (* ?a27 ?b27) (* ?a28 ?b28) (* ?a29 ?b29) (* ?a30 ?b30) (* ?a31 ?b31)), applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8 ?a9 ?a10 ?a11 ?a12 ?a13 ?a14 ?a15 ?a16 ?a17 ?a18 ?a19 ?a20 ?a21 ?a22 ?a23 ?a24 ?a25 ?a26 ?a27 ?a28 ?a29 ?a30 ?a31) (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15 ?b16 ?b17 ?b18 ?b19 ?b20 ?b21 ?b22 ?b23 ?b24 ?b25 ?b26 ?b27 ?b28 ?b29 ?b30 ?b31)) }}
Rewrite {{ name: "rot-mul-vectorize-1", searcher: (Vec (* ?a0 ?b0)), applier: (VecMul (Vec ?a0 ?b0) (<< (Vec ?a0 ?b0) 1)) }}
Rewrite {{ name: "rot-mul-vectorize-2", searcher: (Vec (* ?a0 ?b0) (* ?a1 ?b1)), applier: (VecMul (Vec ?a0 ?a1 ?b0 ?b1) (<< (Vec ?a0 ?a1 ?b0 ?b1) 2)) }}
Rewrite {{ name: "rot-mul-vectorize-4", searcher: (Vec (* ?a0 ?b0) (* ?a1 ?b1) (* ?a2 ?b2) (* ?a3 ?b3)), applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?b0 ?b1 ?b2 ?b3) (<< (Vec ?a0 ?a1 ?a2 ?a3 ?b0 ?b1 ?b2 ?b3) 4)) }}
Rewrite {{ name: "rot-mul-vectorize-8", searcher: (Vec (* ?a0 ?b0) (* ?a1 ?b1) (* ?a2 ?b2) (* ?a3 ?b3) (* ?a4 ?b4) (* ?a5 ?b5) (* ?a6 ?b6) (* ?a7 ?b7)), applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7) (<< (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7) 8)) }}
Rewrite {{ name: "rot-mul-vectorize-16", searcher: (Vec (* ?a0 ?b0) (* ?a1 ?b1) (* ?a2 ?b2) (* ?a3 ?b3) (* ?a4 ?b4) (* ?a5 ?b5) (* ?a6 ?b6) (* ?a7 ?b7) (* ?a8 ?b8) (* ?a9 ?b9) (* ?a10 ?b10) (* ?a11 ?b11) (* ?a12 ?b12) (* ?a13 ?b13) (* ?a14 ?b14) (* ?a15 ?b15)), applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8 ?a9 ?a10 ?a11 ?a12 ?a13 ?a14 ?a15 ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15) (<< (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8 ?a9 ?a10 ?a11 ?a12 ?a13 ?a14 ?a15 ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15) 16)) }}
Rewrite {{ name: "neg-vectorize-1", searcher: (Vec (- ?b0)), applier: (VecNeg (Vec ?b0)) }}
Rewrite {{ name: "neg-vectorize-2", searcher: (Vec (- ?b0) (- ?b1)), applier: (VecNeg (Vec ?b0 ?b1)) }}
Rewrite {{ name: "neg-vectorize-4", searcher: (Vec (- ?b0) (- ?b1) (- ?b2) (- ?b3)), applier: (VecNeg (Vec ?b0 ?b1 ?b2 ?b3)) }}
Rewrite {{ name: "neg-vectorize-8", searcher: (Vec (- ?b0) (- ?b1) (- ?b2) (- ?b3) (- ?b4) (- ?b5) (- ?b6) (- ?b7)), applier: (VecNeg (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7)) }}
Rewrite {{ name: "neg-vectorize-16", searcher: (Vec (- ?b0) (- ?b1) (- ?b2) (- ?b3) (- ?b4) (- ?b5) (- ?b6) (- ?b7) (- ?b8) (- ?b9) (- ?b10) (- ?b11) (- ?b12) (- ?b13) (- ?b14) (- ?b15)), applier: (VecNeg (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15)) }}
Rewrite {{ name: "neg-vectorize-32", searcher: (Vec (- ?b0) (- ?b1) (- ?b2) (- ?b3) (- ?b4) (- ?b5) (- ?b6) (- ?b7) (- ?b8) (- ?b9) (- ?b10) (- ?b11) (- ?b12) (- ?b13) (- ?b14) (- ?b15) (- ?b16) (- ?b17) (- ?b18) (- ?b19) (- ?b20) (- ?b21) (- ?b22) (- ?b23) (- ?b24) (- ?b25) (- ?b26) (- ?b27) (- ?b28) (- ?b29) (- ?b30) (- ?b31)), applier: (VecNeg (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 ?b11 ?b12 ?b13 ?b14 ?b15 ?b16 ?b17 ?b18 ?b19 ?b20 ?b21 ?b22 ?b23 ?b24 ?b25 ?b26 ?b27 ?b28 ?b29 ?b30 ?b31)) }}
Rewrite {{ name: "assoc-balan-add-1", searcher: (VecAdd ?x (VecAdd ?y (VecAdd ?z ?t))), applier: (VecAdd (VecAdd ?x ?y) (VecAdd ?z ?t)) }}
Rewrite {{ name: "assoc-balan-add-2", searcher: (VecAdd ?x (VecAdd (VecAdd ?z ?t) ?y)), applier: (VecAdd (VecAdd ?x ?z) (VecAdd ?t ?y)) }}
Rewrite {{ name: "assoc-balan-add-3", searcher: (VecAdd (VecAdd (VecAdd ?x ?y) ?z) ?t), applier: (VecAdd (VecAdd ?x ?y) (VecAdd ?z ?t)) }}
Rewrite {{ name: "assoc-balan-add-4", searcher: (VecAdd (VecAdd ?x (VecAdd ?y ?z)) ?t), applier: (VecAdd (VecAdd ?x ?y) (VecAdd ?z ?t)) }}
Rewrite {{ name: "assoc-balan-min-1", searcher: (VecMinus ?x (VecMinus ?y (VecMinus ?z ?t))), applier: (VecMinus (VecMinus ?x ?y) (VecMinus ?z ?t)) }}
Rewrite {{ name: "assoc-balan-min-2", searcher: (VecMinus ?x (VecMinus (VecMinus ?z ?t) ?y)), applier: (VecMinus (VecMinus ?x ?z) (VecMinus ?t ?y)) }}
Rewrite {{ name: "assoc-balan-min-3", searcher: (VecMinus (VecMinus (VecMinus ?x ?y) ?z) ?t), applier: (VecMinus (VecMinus ?x ?y) (VecMinus ?z ?t)) }}
Rewrite {{ name: "assoc-balan-min-4", searcher: (VecMinus (VecMinus ?x (VecMinus ?y ?z)) ?t), applier: (VecMinus (VecMinus ?x ?y) (VecMinus ?z ?t)) }}
Rewrite {{ name: "assoc-balan-mul-1", searcher: (VecMul ?x (VecMul ?y (VecMul ?z ?t))), applier: (VecMul (VecMul ?x ?y) (VecMul ?z ?t)) }}
Rewrite {{ name: "assoc-balan-mul-2", searcher: (VecMul ?x (VecMul (VecMul ?z ?t) ?y)), applier: (VecMul (VecMul ?x ?z) (VecMul ?t ?y)) }}
Rewrite {{ name: "assoc-balan-mul-3", searcher: (VecMul (VecMul (VecMul ?x ?y) ?z) ?t), applier: (VecMul (VecMul ?x ?y) (VecMul ?z ?t)) }}
Rewrite {{ name: "assoc-balan-mul-4", searcher: (VecMul (VecMul ?x (VecMul ?y ?z)) ?t), applier: (VecMul (VecMul ?x ?y) (VecMul ?z ?t)) }}
Rewrite {{ name: "assoc-balan-mul-5", searcher: (VecMul ?x (VecMul (VecMul ?y ?z) ?t)), applier: (VecMul (VecMul ?x ?y) (VecMul ?z ?t)) }}
Rewrite {{ name: "assoc-balan-mul-6", searcher: (VecMul ?x (VecMul (VecMul ?y ?z) ?t)), applier: (VecMul (VecMul ?x ?y) (VecMul ?z ?t)) }}
Rewrite {{ name: "assoc-balan-add-mul-1", searcher: (VecAdd (VecAdd (VecAdd (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)) (VecMul ?b1 ?b2)) (VecMul ?a1 ?a2)), applier: (VecAdd (VecAdd (VecMul ?a1 ?a2) (VecMul ?b1 ?b2)) (VecAdd (VecMul ?c1 ?c2) (VecMul ?d1 ?d2))) }}
Rewrite {{ name: "assoc-balan-add-mul-2", searcher: (VecAdd (VecMul ?a1 ?a2) (VecAdd (VecMul ?b1 ?b2) (VecAdd (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)))), applier: (VecAdd (VecAdd (VecMul ?a1 ?a2) (VecMul ?b1 ?b2)) (VecAdd (VecMul ?c1 ?c2) (VecMul ?d1 ?d2))) }}
Rewrite {{ name: "assoc-balan-add-mul-3", searcher: (VecAdd (VecAdd (VecMul ?a1 ?a2) (VecAdd (VecMul ?b1 ?b2) (VecMul ?c1 ?c2))) (VecMul ?d1 ?d2)), applier: (VecAdd (VecAdd (VecMul ?a1 ?a2) (VecMul ?b1 ?b2)) (VecAdd (VecMul ?c1 ?c2) (VecMul ?d1 ?d2))) }}
Rewrite {{ name: "assoc-balan-add-mul-4", searcher: (VecAdd (VecAdd (VecMul ?a ?b) ?c) ?d), applier: (VecAdd (VecMul ?a ?b) (VecAdd ?c ?d)) }}
Rewrite {{ name: "assoc-balan-add-mul-5", searcher: (VecAdd ?a (VecAdd ?b (VecMul ?c ?d))), applier: (VecAdd (VecAdd ?a ?b) (VecMul ?c ?d)) }}
Rewrite {{ name: "distribute-mul-over-add-1", searcher: (VecMul ?a (VecAdd ?b ?c)), applier: (VecAdd (VecMul ?a ?b) (VecMul ?a ?c)) }}
Rewrite {{ name: "assoc-balan-add-min-1", searcher: (VecAdd (VecAdd (VecAdd (VecMinus ?c1 ?c2) (VecMinus ?d1 ?d2)) (VecMinus ?b1 ?b2)) (VecMinus ?a1 ?a2)), applier: (VecAdd (VecAdd (VecMinus ?a1 ?a2) (VecMinus ?b1 ?b2)) (VecAdd (VecMinus ?c1 ?c2) (VecMinus ?d1 ?d2))) }}
Rewrite {{ name: "assoc-balan-add-min-2", searcher: (VecAdd (VecMinus ?a1 ?a2) (VecAdd (VecMinus ?b1 ?b2) (VecAdd (VecMinus ?c1 ?c2) (VecMinus ?d1 ?d2)))), applier: (VecAdd (VecAdd (VecMinus ?a1 ?a2) (VecMinus ?b1 ?b2)) (VecAdd (VecMinus ?c1 ?c2) (VecMinus ?d1 ?d2))) }}
Rewrite {{ name: "assoc-balan-add-min-3", searcher: (VecAdd (VecAdd (VecMinus ?a1 ?a2) (VecAdd (VecMinus ?b1 ?b2) (VecMinus ?c1 ?c2))) (VecMinus ?d1 ?d2)), applier: (VecAdd (VecAdd (VecMinus ?a1 ?a2) (VecMinus ?b1 ?b2)) (VecAdd (VecMinus ?c1 ?c2) (VecMinus ?d1 ?d2))) }}
Rewrite {{ name: "assoc-balan-min-mul-1", searcher: (VecMinus (VecMinus (VecMinus (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)) (VecMul ?b1 ?b2)) (VecMul ?a1 ?a2)), applier: (VecMinus (VecMinus (VecMul ?a1 ?a2) (VecMul ?b1 ?b2)) (VecAdd (VecMul ?c1 ?c2) (VecMul ?d1 ?d2))) }}
Rewrite {{ name: "assoc-balan-min-mul-2", searcher: (VecMinus (VecMul ?a1 ?a2) (VecMinus (VecMul ?b1 ?b2) (VecMinus (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)))), applier: (VecMinus (VecMinus (VecMul ?a1 ?a2) (VecMul ?b1 ?b2)) (VecMinus (VecMul ?c1 ?c2) (VecMul ?d1 ?d2))) }}
Rewrite {{ name: "assoc-balan-min-mul-3", searcher: (VecMinus (VecMinus (VecMul ?a1 ?a2) (VecMinus (VecMul ?b1 ?b2) (VecMul ?c1 ?c2))) (VecMul ?d1 ?d2)), applier: (VecMinus (VecMinus (VecMul ?a1 ?a2) (VecMul ?b1 ?b2)) (VecMinus (VecMul ?c1 ?c2) (VecMul ?d1 ?d2))) }}
Rewrite {{ name: "simplify-sub-negate", searcher: (VecMinus ?x (VecNeg ?y)), applier: (VecAdd ?x ?y) }}
Rewrite {{ name: "simplify-sub-negate-1", searcher: (VecAdd ?x (VecNeg ?y)), applier: (VecMinus ?x ?y) }}
Rewrite {{ name: "simplify-sub-negate-1-2", searcher: (VecAdd (VecNeg ?y) ?x), applier: (VecMinus ?x ?y) }}
Rewrite {{ name: "simplify-add-mul-negate-1", searcher: (VecAdd (VecMul ?x (VecNeg ?y)) ?z), applier: (VecMinus ?z (VecMul ?x ?y)) }}
Rewrite {{ name: "simplify-add-mul-negate-2", searcher: (VecAdd (VecMul (VecNeg ?y) ?x) ?z), applier: (VecMinus ?z (VecMul ?x ?y)) }}
Rewrite {{ name: "simplify-add-mul-negate-3", searcher: (VecAdd ?z (VecMul ?x (VecNeg ?y))), applier: (VecMinus ?z (VecMul ?x ?y)) }}
Rewrite {{ name: "simplify-add-mul-negate-4", searcher: (VecAdd ?z (VecMul (VecNeg ?y) ?x)), applier: (VecMinus ?z (VecMul ?y ?x)) }}
Rewrite {{ name: "simplify-sub-mul-negate-1", searcher: (VecMinus ?z (VecMul ?x (VecNeg ?y))), applier: (VecAdd ?z (VecMul ?x ?y)) }}
Rewrite {{ name: "simplify-sub-mul-negate-2", searcher: (VecMinus ?z (VecMul (VecNeg ?y) ?x)), applier: (VecAdd ?z (VecMul ?x ?y)) }}
Rewrite {{ name: "simplify-add-negate-2-1", searcher: (VecAdd ?x (VecMinus (VecNeg ?y) ?z)), applier: (VecMinus ?x (VecAdd ?x ?y)) }}
Rewrite {{ name: "simplify-add-negate-2-2", searcher: (VecAdd (VecMinus ?z (VecNeg ?y)) ?x), applier: (VecMinus ?x (VecAdd ?x ?y)) }}
Rewrite {{ name: "comm-factor-1", searcher: (+ (* ?a0 ?b0) (* ?a0 ?c0)), applier: (* ?a0 (+ ?b0 ?c0)) }}
Rewrite {{ name: "comm-factor-2", searcher: (+ (* ?b0 ?a0) (* ?c0 ?a0)), applier: (* ?a0 (+ ?b0 ?c0)) }}



---

## Example Expressions (for reference only)

Below are **examples** of existing "vec lang" expressions and their rough computation names.  
**Do not copy or trivially rename them in your new expressions.**  

1. **box_blur** example:
(Vec ( + in_1_1 ( + in_1_0 ( + in_0_1 in_0_0 ) ) ) ( + in_1_2 ( + in_1_1 ( + in_1_0 ( + in_0_2 ( + in_0_1 in_0_0 ) ) ) ) ) ( + in_1_2 ( + in_1_1 ( + in_0_2 in_0_1 ) ) ) ( + in_2_1 ( + in_2_0 ( + in_1_1 ( + in_1_0 ( + in_0_1 in_0_0 ) ) ) ) ) ( + in_2_2 ( + in_2_1 ( + in_2_0 ( + in_1_2 ( + in_1_1 ( + in_1_0 ( + in_0_2 ( + in_0_1 in_0_0 ) ) ) ) ) ) ) ) ( + in_2_2 ( + in_2_1 ( + in_1_2 ( + in_1_1 ( + in_0_2 in_0_1 ) ) ) ) ) ( + in_2_1 ( + in_2_0 ( + in_1_1 in_1_0 ) ) ) ( + in_2_2 ( + in_2_1 ( + in_2_0 ( + in_1_2 ( + in_1_1 in_1_0 ) ) ) ) ) ( + in_2_2 ( + in_2_1 ( + in_1_2 in_1_1 ) ) ) )


2. **dot_product** example:
(Vec ( + ( + ( * v1_0 v2_0 ) ( * v1_1 v2_1 ) ) ( * v1_2 v2_2 ) ) 0 0 0 0 0 0 0 0 )

3. **gx_kernel** example:
(Vec ( + ( * in_0_1 2 ) in_1_1 ) ( + ( + ( + ( * in_0_0 2 ) ( * in_0_2 2 ) ) in_1_0 ) in_1_2 ) ( + in_1_1 ( * in_0_1 2 ) ) ( + ( + in_0_1 ( * in_1_1 2 ) ) in_2_1 ) ( + ( + ( + ( + ( + in_0_0 in_0_2 ) ( * in_1_0 2 ) ) ( * in_1_2 2 ) ) in_2_0 ) in_2_2 ) ( + in_2_1 ( + ( * in_1_1 2 ) in_0_1 ) ) ( + in_1_1 ( * in_2_1 2 ) ) ( + ( + ( + in_1_2 in_1_0 ) ( * in_2_0 2 ) ) ( * in_2_2 2 ) ) ( + ( * in_2_1 2 ) in_1_1 ) )

4. **gy_kernel** example:
(Vec ( + ( * in_1_0 2 ) in_1_1 ) ( + ( + in_1_0 ( * in_1_1 2 ) ) in_1_2 ) ( + in_1_1 ( * in_1_2 2 ) ) ( + ( + ( + ( * in_0_0 -2 ) ( - in_0_1 ) ) ( * in_2_0 2 ) ) in_2_1 ) ( + ( + ( + ( + ( + ( - in_0_0 ) ( * in_0_1 -2 ) ) ( - in_0_2 ) ) in_2_0 ) ( * in_2_1 2 ) ) in_2_2 ) ( + ( + in_2_1 ( + ( - in_0_1 ) ( * in_0_2 -2 ) ) ) ( * in_2_2 2 ) ) ( + ( * in_1_0 -2 ) ( - in_1_1 ) ) ( + ( + ( - in_1_0 ) ( * in_1_1 -2 ) ) ( - in_1_2 ) ) ( + ( - in_1_1 ) ( * in_1_2 -2 ) ) )

5. **hamming_dist** example:
(Vec ( + ( + ( - ( + v1_0 v2_0 ) ( * ( * v1_0 v2_0 ) 2 ) ) ( - ( + v1_1 v2_1 ) ( * 2 ( * v1_1 v2_1 ) ) ) ) ( - ( + v1_2 v2_2 ) ( * 2 ( * v1_2 v2_2 ) ) ) ) 0 0 0 0 0 0 0 0 )

6. **l2_distance** example:
(Vec ( + ( + ( * ( - c2_0 c1_0 ) ( - c2_0 c1_0 ) ) ( * ( - c2_1 c1_1 ) ( - c2_1 c1_1 ) ) ) ( * ( - c2_2 c1_2 ) ( - c2_2 c1_2 ) ) ) 0 0 0 0 0 0 0 0 )

7. **lin_reg** example:
(Vec ( + v2_0 ( + 2 ( * v1_0 5 ) ) ) ( + v2_1 ( + 2 ( * v1_1 5 ) ) ) ( + v2_2 ( + 2 ( * v1_2 5 ) ) ) 0 0 0 0 0 0 )

8. **matrix_mul** example:
(Vec ( + ( + ( * a_0_0 b_0_0 ) ( * a_0_1 b_1_0 ) ) ( * a_0_2 b_2_0 ) ) ( + ( + ( * a_0_0 b_0_1 ) ( * a_0_1 b_1_1 ) ) ( * a_0_2 b_2_1 ) ) ( + ( + ( * a_0_0 b_0_2 ) ( * a_0_1 b_1_2 ) ) ( * a_0_2 b_2_2 ) ) ( + ( + ( * a_1_0 b_0_0 ) ( * a_1_1 b_1_0 ) ) ( * a_1_2 b_2_0 ) ) ( + ( + ( * a_1_0 b_0_1 ) ( * a_1_1 b_1_1 ) ) ( * a_1_2 b_2_1 ) ) ( + ( + ( * a_1_0 b_0_2 ) ( * a_1_1 b_1_2 ) ) ( * a_1_2 b_2_2 ) ) ( + ( + ( * a_2_0 b_0_0 ) ( * a_2_1 b_1_0 ) ) ( * a_2_2 b_2_0 ) ) ( + ( + ( * a_2_0 b_0_1 ) ( * a_2_1 b_1_1 ) ) ( * a_2_2 b_2_1 ) ) ( + ( + ( * a_2_0 b_0_2 ) ( * a_2_1 b_1_2 ) ) ( * a_2_2 b_2_2 ) ) )

9. **max** example:

(Vec ( + ( * ( + ( * ( + ( * o5 ( - 1 c45 ) ) ( * c45 o4 ) ) ( - 1 c34 ) ) ( * c34 ( + ( * o5 ( - 1 c35 ) ) ( * c35 o3 ) ) ) ) ( - 1 c23 ) ) ( * c23 ( + ( * ( + ( * o5 ( - 1 c45 ) ) ( * c45 o4 ) ) ( - 1 c24 ) ) ( * c24 ( + ( * o5 ( - 1 c25 ) ) ( * c25 o2 ) ) ) ) ) ) ( - 1 c12 ) ) ( * c12 ( + ( * ( + ( * ( + ( * o5 ( - 1 c45 ) ) ( * c45 o4 ) ) ( - 1 c34 ) ) ( * c34 ( + ( * o5 ( - 1 c35 ) ) ( * c35 o3 ) ) ) ) ( - 1 c13 ) ) ( * c13 ( + ( * ( + ( * o5 ( - 1 c45 ) ) ( * c45 o4 ) ) ( - 1 c14 ) ) ( * c14 ( + ( * o5 ( - 1 c15 ) ) ( * c15 o1 ) ) ) ) ) ) ) 0 0 0 0 0 0 0 0 )

10. **polynomials_coyote** example:
(Vec ( + ( + x x ) ( * x x ) ) 0 0 0 0 0 0 0 0 )

11. **poly_reg** example:
(Vec ( + c1_0 ( + c2_0 ( + ( * c0_0 c3_0 ) ( * c0_0 ( * c0_0 c4_0 ) ) ) ) ) ( + c1_1 ( + c2_1 ( + ( * c0_1 c3_1 ) ( * c0_1 ( * c0_1 c4_1 ) ) ) ) ) ( + c1_2 ( + c2_2 ( + ( * c0_2 c3_2 ) ( * c0_2 ( * c0_2 c4_2 ) ) ) ) ) 0 0 0 0 0 0 )

12. **roberts_cross** example:
(Vec ( + ( * ( + ( - in_1_0 ) in_0_1 ) ( + ( - in_1_0 ) in_0_1 ) ) ( * ( + ( - in_1_1 ) in_0_2 ) ( + ( - in_1_1 ) in_0_2 ) ) ) ( + ( * ( + ( - in_1_1 ) in_0_2 ) ( + ( - in_1_1 ) in_0_2 ) ) ( * ( + ( - in_1_2 ) in_0_1 ) ( + ( - in_1_2 ) in_0_1 ) ) ) ( + ( * ( - in_1_2 ) ( - in_1_2 ) ) ( * in_0_2 in_0_2 ) ) ( + ( * ( + ( - in_2_0 ) in_1_1 ) ( + ( - in_2_0 ) in_1_1 ) ) ( * ( + ( - in_2_1 ) in_1_0 ) ( + ( - in_2_1 ) in_1_0 ) ) ) ( + ( * ( + ( - in_2_1 ) in_1_2 ) ( + ( - in_2_1 ) in_1_2 ) ) ( * ( + ( - in_2_2 ) in_1_1 ) ( + ( - in_2_2 ) in_1_1 ) ) ) ( + ( * ( - in_2_2 ) ( - in_2_2 ) ) ( * in_1_2 in_1_2 ) ) ( + ( * in_2_1 in_2_1 ) ( * in_2_0 in_2_0 ) ) ( + ( * in_2_2 in_2_2 ) ( * in_2_1 in_2_1 ) ) ( * in_2_2 in_2_2 ) )

13. **sobel** example:
(Vec ( + ( * ( + in_1_1 ( * in_1_0 2 ) ) ( + in_1_1 ( * in_1_0 2 ) ) ) ( * ( + ( * in_0_1 2 ) in_1_1 ) ( + ( * in_0_1 2 ) in_1_1 ) ) ) ( + ( * ( + in_1_2 ( + ( * in_1_1 2 ) in_1_0 ) ) ( + in_1_2 ( + ( * in_1_1 2 ) in_1_0 ) ) ) ( * ( + ( + ( + ( * in_0_0 -2 ) ( * in_0_2 2 ) ) ( - in_1_0 ) ) in_1_2 ) ( + ( + ( + ( * in_0_0 -2 ) ( * in_0_2 2 ) ) ( - in_1_0 ) ) in_1_2 ) ) ) ( + ( * ( + ( * in_1_2 2 ) in_1_1 ) ( + ( * in_1_2 2 ) in_1_1 ) ) ( * ( + ( * in_0_1 -2 ) ( - in_1_1 ) ) ( + ( * in_0_1 -2 ) ( - in_1_1 ) ) ) ) ( + ( * ( + in_2_1 ( + ( + ( * in_0_0 -2 ) ( - in_0_1 ) ) ( * in_2_0 2 ) ) ) ( + in_2_1 ( + ( + ( * in_0_0 -2 ) ( - in_0_1 ) ) ( * in_2_0 2 ) ) ) ) ( * ( + ( + in_0_1 ( * in_1_1 2 ) ) in_2_1 ) ( + ( + in_0_1 ( * in_1_1 2 ) ) in_2_1 ) ) ) ( + ( * ( + in_2_2 ( + ( * in_2_1 2 ) ( + ( + ( + ( * in_0_1 -2 ) ( - in_0_0 ) ) ( - in_0_2 ) ) in_2_0 ) ) ) ( + in_2_2 ( + ( * in_2_1 2 ) ( + ( + ( + ( * in_0_1 -2 ) ( - in_0_0 ) ) ( - in_0_2 ) ) in_2_0 ) ) ) ) ( * ( + ( + ( + ( + ( + ( - in_0_0 ) in_0_2 ) ( * in_1_0 -2 ) ) ( * in_1_2 2 ) ) ( - in_2_0 ) ) in_2_2 ) ( + ( + ( + ( + ( + ( - in_0_0 ) in_0_2 ) ( * in_1_0 -2 ) ) ( * in_1_2 2 ) ) ( - in_2_0 ) ) in_2_2 ) ) ) ( + ( * ( + ( * in_2_2 2 ) ( + in_2_1 ( + ( - in_0_1 ) ( * in_0_2 -2 ) ) ) ) ( + ( * in_2_2 2 ) ( + in_2_1 ( + ( - in_0_1 ) ( * in_0_2 -2 ) ) ) ) ) ( * ( + ( + ( - in_0_1 ) ( * in_1_1 -2 ) ) ( - in_2_1 ) ) ( + ( + ( - in_0_1 ) ( * in_1_1 -2 ) ) ( - in_2_1 ) ) ) ) ( + ( * ( + ( - in_1_1 ) ( * in_1_0 -2 ) ) ( + ( - in_1_1 ) ( * in_1_0 -2 ) ) ) ( * ( + in_1_1 ( * in_2_1 2 ) ) ( + in_1_1 ( * in_2_1 2 ) ) ) ) ( + ( * ( + ( + ( * in_1_1 -2 ) ( - in_1_0 ) ) ( - in_1_2 ) ) ( + ( + ( * in_1_1 -2 ) ( - in_1_0 ) ) ( - in_1_2 ) ) ) ( * ( + ( + ( + in_1_2 ( - in_1_0 ) ) ( * in_2_0 -2 ) ) ( * in_2_2 2 ) ) ( + ( + ( + in_1_2 ( - in_1_0 ) ) ( * in_2_0 -2 ) ) ( * in_2_2 2 ) ) ) ) ( + ( * ( + ( - in_1_1 ) ( * in_1_2 -2 ) ) ( + ( - in_1_1 ) ( * in_1_2 -2 ) ) ) ( * ( + ( - in_1_1 ) ( * in_2_1 -2 ) ) ( + ( - in_1_1 ) ( * in_2_1 -2 ) ) ) ) )

**These examples are here for inspiration only.**  
**Do not reproduce or trivially rename them.**

---


**Generate 3 NEW expressions meeting ALL criteria**:

 1- Use mixed variables/constants

 2- Unique structure (not isomorphic to examples or the ones already generated)
 3- Validate parentheses before output

 4- One expression per line
 
 5- No styling for output
 
 6- Never Generate Expressions that have 0 in them as sub expression
 
 7 - Generate only expression where a sequence of rules can be applied to them
 
 8 - Never Generate expressions that have constants only always generate expressions that can be applied rules to it
 
 
 11 - **Do not generate expression that exists in examples by just changing variable names**

 12 - All operations have only 2 children no more
 
 13 - Generate Only Vec expressions Do Not use Vector operations shuch as VecMul and VecNeg and VecAdd only use Vec with scalar opeartaion
 
 14 - Generate **Unique expressions** do not generate the same expressions over and over each time generate unique expressions

 15 - **Structural Originality**: Expressions must differ in operator topology/nesting patterns, not just variable names. 
     Example: (Vec (+ a b) c) vs (Vec (+ x y) z) = SAME STRUCTURE = INVALID 
              - (Vec (+ a b) (- c d) ...) → Same structure as (Vec (+ x y) (- z w) ...) → Invalid duplicate

 16 - Generate Somtimes deep nested expressions with very high depth
 
 17 - **Never generate a sub expressions with parantheses with only one variable or constant example (0) or (v_0_1) these are not to done**
 
 
 18 - Generate expressions with depth more then 6
**Output ONLY valid VecLang expressions**
** Output Only valid expressions Ignore the expressions that does not match our requirement**
** Generate expressions from simple to complex (Low depth and pretty simple to complex structure expressions)**
** Output Should be clean expressions without any addition to them or any stilying just raw expression **
** Generate expressions where there multiplicative depth and depth can be reduced if a sequence of rules are applied**
** Each output expression should have a unique structure with distinct computation graphs (rathen then changing only variable names)**



**Generate expressions with distinct computation graphs** - Two expressions differ if:
1. Operator nesting hierarchy differs (e.g., (+ (* a b) c) vs (* (+ a b) c))
2. Operation distribution differs (e.g., (+ a (+ b c)) vs (+ (+ a b) c))
3. Constant/variable positioning pattern differs (e.g., (+ C V) vs (+ V C) allowed ONLY if all instances show non-symmetric patterns)

**Output ONLY 5 structurally unique VecLang expressions meeting ALL criteria**
** Output should only contain the expressions without any additional data or Text or styling just raw outputs**
** Output each expression on it's own line** Do No

**Generate only 5 structurally unique VecLang expressions**, each with `{vec_size}` sub-expressions**
** I want the generated expressions to be simple and be not completly vectorizable for example Vec ( (+ a b) (+ c d) (- f g)) so that the RL agent can learn to optimise it and make it vectorizable**
"""