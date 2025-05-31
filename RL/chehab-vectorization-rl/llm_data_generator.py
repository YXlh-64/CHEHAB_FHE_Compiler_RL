from langchain_google_genai import GoogleGenerativeAI

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import time
from langchain.schema import StrOutputParser

chat_history = []

with open("vec_lang_expressions.txt", "r") as file:
    chat_history.append("\n".join(file.readlines()))


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a data generator assistant for VecLang.",
        ),
        ("human", """
Below is a domain-specific language called **"vec lang"**, which has the following characteristics:

1. Top-level structure starts with **(Vec <subexpr1> <subexpr2> ... <subexpr9>)**.
2. Operators: **+, -, *** (never /).
3. Variables: They look like **in_0_0, in_1_1, c1_0, o5, v1_0,** etc.
4. Grammar-like rules (simplified):
   - An expression can be:
     - (**<op>** **<expr>** **<expr>**)
     - A variable (like in_0_0)
     - A constant or function call in parentheses
   - The **<op>** can be **+, -, ***.
   - **Each (Vec ...) must have exactly 9 sub-expressions** (9 "slots").
   - **Nesting depth** of each expression must be **at least 4** and **at most 11**.
   - ** Ensure that all generated expressions have strictly balanced and correctly placed parentheses.No missing or extra parentheses are allowed, and each (Vec ...) must parse without any syntax errors.**

---
Here is our previous history (expressions we must **not** duplicate):
{history}
## Rewriting Rules (for context only)

Below are rewriting rules for "vec lang." Note that they are wrapped in double braces `{{ ... }}` so they are not interpreted as template variables. **Do not output these rules in your answer.** They are here so you know how expressions might be rewritten multiple times.

Rewrite {{ name: "add-0-0+0", searcher: 0, applier: (+ 0 0) }}
Rewrite {{ name: "mul-1-1x1", searcher: 1, applier: (* 1 1) }}
Rewrite {{ name: "add-0", searcher: (+ 0 ?a), applier: ?a }}
Rewrite {{ name: "add-0-2", searcher: (+ ?a 0), applier: ?a }}
Rewrite {{ name: "mul-0", searcher: (* 0 ?a), applier: 0 }}
Rewrite {{ name: "mul-0-2", searcher: (* ?a 0), applier: 0 }}
Rewrite {{ name: "mul-1", searcher: (* 1 ?a), applier: ?a }}
Rewrite {{ name: "mul-1-2", searcher: (* ?a 1), applier: ?a }}
Rewrite {{ name: "comm-factor-1", searcher: (+ (* ?a0 ?b0) (* ?a0 ?c0)), applier: (* ?a0 (+ ?b0 ?c0)) }}
Rewrite {{ name: "comm-factor-2", searcher: (+ (* ?b0 ?a0) (* ?c0 ?a0)), applier: (* ?a0 (+ ?b0 ?c0)) }}
Rewrite {{ name: "add-vectorize",
  searcher: (Vec (+ ?a0 ?b0) (+ ?a1 ?b1) (+ ?a2 ?b2) (+ ?a3 ?b3) (+ ?a4 ?b4) (+ ?a5 ?b5) (+ ?a6 ?b6) (+ ?a7 ?b7) (+ ?a8 ?b8)),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8))
}}
Rewrite {{ name: "mul-vectorize",
  searcher: (Vec (* ?a0 ?b0) (* ?a1 ?b1) (* ?a2 ?b2) (* ?a3 ?b3) (* ?a4 ?b4) (* ?a5 ?b5) (* ?a6 ?b6) (* ?a7 ?b7) (* ?a8 ?b8)),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8))
}}
Rewrite {{ name: "sub-vectorize",
  searcher: (Vec (- ?a0 ?b0) (- ?a1 ?b1) (- ?a2 ?b2) (- ?a3 ?b3) (- ?a4 ?b4) (- ?a5 ?b5) (- ?a6 ?b6) (- ?a7 ?b7) (- ?a8 ?b8)),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec ?b0 ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8))
}}
Rewrite {{ name: "neg-vectorize",
  searcher: (Vec (- ?a0) (- ?a1) (- ?a2) (- ?a3) (- ?a4) (- ?a5) (- ?a6) (- ?a7) (- ?a8)),
  applier: (VecNeg (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8))
}}
Rewrite {{ name: "add-split-0",
  searcher: (Vec (+ ?a01 ?a02) ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a01 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec ?a02 0 0 0 0 0 0 0 0))
}}
Rewrite {{ name: "mul-split-0",
  searcher: (Vec (* ?a01 ?a02) ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a01 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec ?a02 1 1 1 1 1 1 1 1))
}}
Rewrite {{ name: "sub-split-0",
  searcher: (Vec (- ?a01 ?a02) ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMinus (Vec ?a01 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec ?a02 0 0 0 0 0 0 0 0))
}}
Rewrite {{ name: "neg-split-0",
  searcher: (Vec (- ?a0) ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMinus (Vec 0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec ?a0 0 0 0 0 0 0 0 0))
}}
Rewrite {{ name: "add-split-1",
  searcher: (Vec ?a0 (+ ?a11 ?a12) ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a11 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 0 ?a12 0 0 0 0 0 0 0))
}}
Rewrite {{ name: "mul-split-1",
  searcher: (Vec ?a0 (* ?a11 ?a12) ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a11 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 1 ?a12 1 1 1 1 1 1 1))
}}
Rewrite {{ name: "sub-split-1",
  searcher: (Vec ?a0 (- ?a11 ?a12) ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMinus (Vec ?a0 ?a11 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 0 ?a12 0 0 0 0 0 0 0))
}}
Rewrite {{ name: "neg-split-1",
  searcher: (Vec ?a0 (- ?a1) ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMinus (Vec ?a0 0 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 0 ?a1 0 0 0 0 0 0 0))
}}
Rewrite {{ name: "add-split-2",
  searcher: (Vec ?a0 ?a1 (+ ?a21 ?a22) ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 ?a21 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 0 0 ?a22 0 0 0 0 0 0))
}}
Rewrite {{ name: "mul-split-2",
  searcher: (Vec ?a0 ?a1 (* ?a21 ?a22) ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a1 ?a21 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 1 1 ?a22 1 1 1 1 1 1))
}}
Rewrite {{ name: "sub-split-2",
  searcher: (Vec ?a0 ?a1 (- ?a21 ?a22) ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMinus (Vec ?a0 ?a1 ?a21 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 0 0 ?a22 0 0 0 0 0 0))
}}
Rewrite {{ name: "neg-split-2",
  searcher: (Vec ?a0 ?a1 (- ?a2) ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMinus (Vec ?a0 ?a1 0 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 0 0 ?a2 0 0 0 0 0 0))
}}
Rewrite {{ name: "add-split-3",
  searcher: (Vec ?a0 ?a1 ?a2 (+ ?a31 ?a32) ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a31 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 0 0 0 ?a32 0 0 0 0 0))
}}
Rewrite {{ name: "mul-split-3",
  searcher: (Vec ?a0 ?a1 ?a2 (* ?a31 ?a32) ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a31 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 1 1 1 ?a32 1 1 1 1 1))
}}
Rewrite {{ name: "sub-split-3",
  searcher: (Vec ?a0 ?a1 ?a2 (- ?a31 ?a32) ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a31 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 0 0 0 ?a32 0 0 0 0 0))
}}
Rewrite {{ name: "neg-split-3",
  searcher: (Vec ?a0 ?a1 ?a2 (- ?a3) ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 0 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 0 0 0 ?a3 0 0 0 0 0))
}}
Rewrite {{ name: "add-split-4",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 (+ ?a41 ?a42) ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a41 ?a5 ?a6 ?a7 ?a8) (Vec 0 0 0 0 ?a42 0 0 0 0))
}}
Rewrite {{ name: "mul-split-4",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 (* ?a41 ?a42) ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a41 ?a5 ?a6 ?a7 ?a8) (Vec 1 1 1 1 ?a42 1 1 1 1))
}}
Rewrite {{ name: "sub-split-4",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 (- ?a41 ?a42) ?a5 ?a6 ?a7 ?a8),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a41 ?a5 ?a6 ?a7 ?a8) (Vec 0 0 0 0 ?a42 0 0 0 0))
}}
Rewrite {{ name: "neg-split-4",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 (- ?a4) ?a5 ?a6 ?a7 ?a8),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 0 ?a5 ?a6 ?a7 ?a8) (Vec 0 0 0 0 ?a4 0 0 0 0))
}}
Rewrite {{ name: "add-split-5",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 (+ ?a51 ?a52) ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a51 ?a6 ?a7 ?a8) (Vec 0 0 0 0 0 ?a52 0 0 0))
}}
Rewrite {{ name: "mul-split-5",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 (* ?a51 ?a52) ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a51 ?a6 ?a7 ?a8) (Vec 1 1 1 1 1 ?a52 1 1 1))
}}
Rewrite {{ name: "sub-split-5",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 (- ?a51 ?a52) ?a6 ?a7 ?a8),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a51 ?a6 ?a7 ?a8) (Vec 0 0 0 0 0 ?a52 0 0 0))
}}
Rewrite {{ name: "neg-split-5",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 (- ?a5) ?a6 ?a7 ?a8),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 0 ?a6 ?a7 ?a8) (Vec 0 0 0 0 0 ?a5 0 0 0))
}}
Rewrite {{ name: "add-split-6",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 (+ ?a61 ?a62) ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a61 ?a7 ?a8) (Vec 0 0 0 0 0 0 ?a62 0 0))
}}
Rewrite {{ name: "mul-split-6",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 (* ?a61 ?a62) ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a61 ?a7 ?a8) (Vec 1 1 1 1 1 1 ?a62 1 1))
}}
Rewrite {{ name: "sub-split-6",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 (- ?a61 ?a62) ?a7 ?a8),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a61 ?a7 ?a8) (Vec 0 0 0 0 0 0 ?a62 0 0))
}}
Rewrite {{ name: "neg-split-6",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 (- ?a6) ?a7 ?a8),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 0 ?a7 ?a8) (Vec 0 0 0 0 0 0 ?a6 0 0))
}}
Rewrite {{ name: "add-split-7",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 (+ ?a71 ?a72) ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a71 ?a8) (Vec 0 0 0 0 0 0 0 ?a72 0))
}}
Rewrite {{ name: "mul-split-7",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 (* ?a71 ?a72) ?a8),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a71 ?a8) (Vec 1 1 1 1 1 1 1 ?a72 1))
}}
Rewrite {{ name: "sub-split-7",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 (- ?a71 ?a72) ?a8),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a71 ?a8) (Vec 0 0 0 0 0 0 0 ?a72 0))
}}
Rewrite {{ name: "neg-split-7",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 (- ?a7) ?a8),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 0 ?a8) (Vec 0 0 0 0 0 0 0 ?a7 0))
}}
Rewrite {{ name: "add-split-8",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 (+ ?a81 ?a82)),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a81) (Vec 0 0 0 0 0 0 0 0 ?a82))
}}
Rewrite {{ name: "mul-split-8",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 (* ?a81 ?a82)),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a81) (Vec 1 1 1 1 1 1 1 1 ?a82))
}}
Rewrite {{ name: "sub-split-8",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 (- ?a81 ?a82)),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a81) (Vec 0 0 0 0 0 0 0 0 ?a82))
}}
Rewrite {{ name: "neg-split-8",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 (- ?a8)),
  applier: (VecMinus (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 0) (Vec 0 0 0 0 0 0 0 0 ?a8))
}}

Rewrite {{ name: "exp-split-add-0",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec 0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec ?a0 0 0 0 0 0 0 0 0))
}}
Rewrite {{ name: "exp-split-mul-0",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec 1 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec ?a0 1 1 1 1 1 1 1 1))
}}
Rewrite {{ name: "exp-split-add-1",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 0 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 0 ?a1 0 0 0 0 0 0 0))
}}
Rewrite {{ name: "exp-split-mul-1",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 1 ?a1 1 1 1 1 1 1 1))
}}
Rewrite {{ name: "exp-split-add-2",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 0 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 0 0 ?a2 0 0 0 0 0 0))
}}
Rewrite {{ name: "exp-split-mul-2",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a1 1 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 1 1 ?a2 1 1 1 1 1 1))
}}
Rewrite {{ name: "exp-split-add-3",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 0 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 0 0 0 ?a3 0 0 0 0 0))
}}
Rewrite {{ name: "exp-split-mul-3",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 1 ?a4 ?a5 ?a6 ?a7 ?a8) (Vec 1 1 1 ?a3 1 1 1 1 1))
}}
Rewrite {{ name: "exp-split-add-4",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 0 ?a5 ?a6 ?a7 ?a8) (Vec 0 0 0 0 ?a4 0 0 0 0))
}}
Rewrite {{ name: "exp-split-mul-4",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 1 ?a5 ?a6 ?a7 ?a8) (Vec 1 1 1 1 ?a4 1 1 1 1))
}}
Rewrite {{ name: "exp-split-add-5",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 0 ?a6 ?a7 ?a8) (Vec 0 0 0 0 0 ?a5 0 0 0))
}}
Rewrite {{ name: "exp-split-mul-5",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 1 ?a6 ?a7 ?a8) (Vec 1 1 1 1 1 ?a5 1 1 1))
}}
Rewrite {{ name: "exp-split-add-6",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 0 ?a7 ?a8) (Vec 0 0 0 0 0 0 ?a6 0 0))
}}
Rewrite {{ name: "exp-split-mul-6",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 1 ?a7 ?a8) (Vec 1 1 1 1 1 1 ?a6 1 1))
}}
Rewrite {{ name: "exp-split-add-7",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 0 ?a8) (Vec 0 0 0 0 0 0 0 ?a7 0))
}}
Rewrite {{ name: "exp-split-mul-7",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 1 ?a8) (Vec 1 1 1 1 1 1 1 ?a7 1))
}}
Rewrite {{ name: "exp-split-add-8",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecAdd (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 0) (Vec 0 0 0 0 0 0 0 0 ?a8))
}}
Rewrite {{ name: "exp-split-mul-8",
  searcher: (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 ?a8),
  applier: (VecMul (Vec ?a0 ?a1 ?a2 ?a3 ?a4 ?a5 ?a6 ?a7 1) (Vec 1 1 1 1 1 1 1 1 ?a8))
}}

Rewrite {{ name: "exp-assoc-0", searcher: (+ (* a0 b0) c0), applier: (+ c0 (* a0 b0)) }}
Rewrite {{ name: "exp-assoc-2", searcher: (+ (* a2 b2) c2), applier: (+ c2 (* a2 b2)) }}
Rewrite {{ name: "exp-assoc-4", searcher: (+ (* a4 b4) c4), applier: (+ c4 (* a4 b4)) }}
Rewrite {{ name: "exp-assoc-6", searcher: (+ (* a6 b6) c6), applier: (+ c6 (* a6 b6)) }}
Rewrite {{ name: "exp-assoc-8", searcher: (+ (* a8 b8) c8), applier: (+ c8 (* a8 b8)) }}
Rewrite {{ name: "assoc-balan-add-1", searcher: (VecAdd ?x (VecAdd ?y (VecAdd ?z ?t))), applier: (VecAdd (VecAdd ?x ?y) (VecAdd ?z ?t)) }}
Rewrite {{ name: "assoc-balan-add-2", searcher: (VecAdd ?x (VecAdd (VecAdd ?z ?t) ?y)), applier: (VecAdd (VecAdd ?x ?z) (VecAdd ?t ?y)) }}
Rewrite {{ name: "assoc-balan-add-3", searcher: (VecAdd (VecAdd (VecAdd ?x ?y) ?z) ?t)), applier: (VecAdd (VecAdd ?x ?y) (VecAdd ?z ?t)) }}
Rewrite {{ name: "assoc-balan-add-4", searcher: (VecAdd (VecAdd ?x (VecAdd ?y ?z)) ?t)), applier: (VecAdd (VecAdd ?x ?y) (VecAdd ?z ?t)) }}
Rewrite {{ name: "assoc-balan-mul-1", searcher: (VecMul ?x (VecMul ?y (VecMul ?z ?t))), applier: (VecMul (VecMul ?x ?y) (VecMul ?z ?t)) }}
Rewrite {{ name: "assoc-balan-mul-2", searcher: (VecMul ?x (VecMul (VecMul ?z ?t) ?y)), applier: (VecMul (VecMul ?x ?z) (VecMul ?t ?y)) }}
Rewrite {{ name: "assoc-balan-mul-3", searcher: (VecMul (VecMul (VecMul ?x ?y) ?z) ?t)), applier: (VecMul (VecMul ?x ?y) (VecMul ?z ?t)) }}
Rewrite {{ name: "assoc-balan-mul-4", searcher: (VecMul (VecMul ?x (VecMul ?y ?z)) ?t)), applier: (VecMul (VecMul ?x ?y) (VecMul ?z ?t)) }}
Rewrite {{ name: "assoc-balan-mul-5", searcher: (VecMul ?x (VecMul (VecMul ?y ?z) ?t)), applier: (VecMul (VecMul ?x ?y) (VecMul ?z ?t)) }}
Rewrite {{ name: "assoc-balan-mul-6", searcher: (VecMul ?x (VecMul (VecMul ?y ?z) ?t)), applier: (VecMul (VecMul ?x ?y) (VecMul ?z ?t)) }}
Rewrite {{ name: "assoc-balan-min-1", searcher: (VecMinus ?x (VecMinus ?y (VecMinus ?z ?t))), applier: (VecMinus (VecMinus ?x ?y) (VecMinus ?z ?t)) }}
Rewrite {{ name: "assoc-balan-min-2", searcher: (VecMinus ?x (VecMinus (VecMinus ?z ?t) ?y)), applier: (VecMinus (VecMinus ?x ?z) (VecMinus ?t ?y)) }}
Rewrite {{ name: "assoc-balan-min-3", searcher: (VecMinus (VecMinus (VecMinus ?x ?y) ?z) ?t)), applier: (VecMinus (VecMinus ?x ?y) (VecMinus ?z ?t)) }}
Rewrite {{ name: "assoc-balan-min-4", searcher: (VecMinus (VecMinus ?x (VecMinus ?y ?z)) ?t)), applier: (VecMinus (VecMinus ?x ?y) (VecMinus ?z ?t)) }}
Rewrite {{ name: "assoc-balan-add-mul-1",
  searcher: (VecAdd (VecAdd (VecAdd (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)) (VecMul ?b1 ?b2)) (VecMul ?a1 ?a2)),
  applier: (VecAdd (VecAdd (VecMul ?a1 ?a2) (VecMul ?b1 ?b2)) (VecAdd (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)))
}}
Rewrite {{ name: "assoc-balan-add-mul-2",
  searcher: (VecAdd (VecMul ?a1 ?a2) (VecAdd (VecMul ?b1 ?b2) (VecAdd (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)))),
  applier: (VecAdd (VecAdd (VecMul ?a1 ?a2) (VecMul ?b1 ?b2)) (VecAdd (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)))
}}
Rewrite {{ name: "assoc-balan-add-mul-3",
  searcher: (VecAdd (VecAdd (VecMul ?a1 ?a2) (VecAdd (VecMul ?b1 ?b2) (VecMul ?c1 ?c2))) (VecMul ?d1 ?d2)),
  applier: (VecAdd (VecAdd (VecMul ?a1 ?a2) (VecMul ?b1 ?b2)) (VecAdd (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)))
}}
Rewrite {{ name: "distribute-mul-over-add",
  searcher: (VecMul ?a (VecAdd ?b ?c)),
  applier: (VecAdd (VecMul ?a ?b) (VecMul ?a ?c))
}}
Rewrite {{ name: "factor-out-mul",
  searcher: (VecAdd (VecMul ?a ?b) (VecMul ?a ?c)),
  applier: (VecMul ?a (VecAdd ?b ?c))
}}
Rewrite {{ name: "assoc-balan-add-min-1",
  searcher: (VecAdd (VecAdd (VecAdd (VecMinus ?c1 ?c2) (VecMinus ?d1 ?d2)) (VecMinus ?b1 ?b2)) (VecMinus ?a1 ?a2)),
  applier: (VecAdd (VecAdd (VecMinus ?a1 ?a2) (VecMinus ?b1 ?b2)) (VecAdd (VecMinus ?c1 ?c2) (VecMinus ?d1 ?d2)))
}}
Rewrite {{ name: "assoc-balan-add-min-2",
  searcher: (VecAdd (VecMinus ?a1 ?a2) (VecAdd (VecMinus ?b1 ?b2) (VecAdd (VecMinus ?c1 ?c2) (VecMinus ?d1 ?d2)))),
  applier: (VecAdd (VecAdd (VecMinus ?a1 ?a2) (VecMinus ?b1 ?b2)) (VecAdd (VecMinus ?c1 ?c2) (VecMinus ?d1 ?d2)))
}}
Rewrite {{ name: "assoc-balan-add-min-3",
  searcher: (VecAdd (VecAdd (VecMinus ?a1 ?a2) (VecAdd (VecMinus ?b1 ?b2) (VecMinus ?c1 ?c2))) (VecMinus ?d1 ?d2)),
  applier: (VecAdd (VecAdd (VecMinus ?a1 ?a2) (VecMinus ?b1 ?b2)) (VecAdd (VecMinus ?c1 ?c2) (VecMinus ?d1 ?d2)))
}}
Rewrite {{ name: "assoc-balan-min-mul-1",
  searcher: (VecMinus (VecMinus (VecMinus (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)) (VecMul ?b1 ?b2)) (VecMul ?a1 ?a2)),
  applier: (VecMinus (VecMinus (VecMul ?a1 ?a2) (VecMul ?b1 ?b2)) (VecAdd (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)))
}}
Rewrite {{ name: "assoc-balan-min-mul-2",
  searcher: (VecMinus (VecMul ?a1 ?a2) (VecMinus (VecMul ?b1 ?b2) (VecMinus (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)))),
  applier: (VecMinus (VecMinus (VecMul ?a1 ?a2) (VecMul ?b1 ?b2)) (VecMinus (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)))
}}
Rewrite {{ name: "assoc-balan-min-mul-3",
  searcher: (VecMinus (VecMinus (VecMul ?a1 ?a2) (VecMinus (VecMul ?b1 ?b2) (VecMul ?c1 ?c2))) (VecMul ?d1 ?d2)),
  applier: (VecMinus (VecMinus (VecMul ?a1 ?a2) (VecMul ?b1 ?b2)) (VecMinus (VecMul ?c1 ?c2) (VecMul ?d1 ?d2)))
}}
Rewrite {{ name: "distribute-mul-over-min",
  searcher: (VecMul ?a (VecMinus ?b ?c)),
  applier: (VecMinus (VecMul ?a ?b) (VecMul ?a ?c))
}}
Rewrite {{ name: "factor-out-mul_min",
  searcher: (VecMinus (VecMul ?a ?b) (VecMul ?a ?c)),
  applier: (VecMul ?a (VecMinus ?b ?c))
}}

---

## Example Expressions (for reference only)

Below are **examples** of existing "vec lang" expressions and their rough computation names.  
**Do not copy or trivially rename them in your new expressions.**  
**Do not include any `|||` suffix or these example names** in your final output.

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

## Your Task

- **Generate 10 new, distinct "vec lang" expressions** that:
  - Each `(Vec ...)` has exactly **9 sub-expressions**.
  - Confirm `(Vec ...)` has exactly **9 sub-expressions** and make sur the parantheses are placed in the right place also never generate bad expressions or with less then 9 sub expressions inside the Vec parent.
  - Uses **only** the operators `+`, `-`, `*` (no `/`).
  - Has a **nesting depth** of at least **4** and at most **11**.
  - **Never** duplicates or trivially renames any expression from the history or from these examples.
  - **Never** generate expressions that exists in  history section means that they were generated before.
  - **Always** generate new and unique expressions never duplicate the generated expressions never never.
  - **Always** the expressions should be unique and never generated before.
  - Each sub-expression should be a “meaningful” part of some known computation (so that rewriting rules can be applied many times).
  - Output **only** the raw `(Vec ...)` expressions, **one per line**, with **no** explanation or additional text.
  - Generate a valid expressions where you need to confirm that all the parantheses are closed and placed correctly .
  - ** Do Not generate invalid expressions with missing parantheses or elements only generate valid expressions **
  - ** Ensure that all generated expressions have strictly balanced and correctly placed parentheses.No missing or extra parentheses are allowed, and each (Vec ...) must parse without any syntax errors.**
  - ** Do not regenerate the same expression by only changing the variable names the strcuture also should be changed **
  - ** The output should be a list of expressions without any additions or formating**
"""),
    ]
)
#llm = GoogleGenerativeAI(model="models/gemini-1.5-pro", google_api_key="AIzaSyB8q8eFa4tIt2hb_BuKGeelBufBzA4XbRQ")
llm = ChatOpenAI(model="gpt-4o", temperature=0.5, api_key="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")
output_parser = StrOutputParser()



chain = prompt | llm | output_parser
with open("vec_lang_expressions.txt", "a") as file:
    for i in tqdm(range(100)):
        history_str = "\n".join(chat_history)
        response = chain.invoke({"history": history_str})
        chat_history.append(response)
        file.write(response)
        file.flush()

