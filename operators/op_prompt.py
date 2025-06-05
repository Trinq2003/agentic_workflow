from prompt.few_shot import FewShotPrompt

SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.
"""

PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
"""

SELFREFINE_PROMPT = """
You are an assistant specialized in refining solutions to problems.

Problem:
{problem}

Solution:
{solution}

Instruction:
Analyze the above solution for any errors or suboptimal aspects. Make iterative improvements to enhance its correctness and efficiency. Provide the refined solution below.
"""

COT_DEMONSTRATION_SAMPLING = FewShotPrompt(
   [
      {
         "role": "user",
         "content": "Problem: Find all real solutions to $x^4 - 3x^2 - 4 = 0$"
      },
      {
         "role": "assistant",
         "content": "Analysis: \nLet $y = x^2$, transforming to quadratic equation:  \n$y^2 - 3y - 4 = 0$  \nFactorization: $(y-4)(y+1)=0$ → $y=4$ or $y=-1$  \nSince $y=x^2 \\geq 0$, discard $y=-1$  \nSolve $x^2=4$ → $x=\\pm 2$  \n\\boxed{-2}, \\boxed{2}"
      },
      {
         "role": "user",
         "content": "Problem: Cone with height 12cm, radius 6cm. Find volume."
      },
      {
         "role": "assistant",
         "content": "Analysis:\nVolume formula: $V = \\frac{1}{3}\\pi r^2 h$  \nSubstitute values: $V = \\frac{1}{3}\\pi(6)^2(12)$  \nCalculate: $6^2=36$ → $36\\times12=432$  \nFinal volume: $\\frac{432}{3}\\pi = 144\\pi$  \n\\boxed{144\\pi}"
      }
   ]
)