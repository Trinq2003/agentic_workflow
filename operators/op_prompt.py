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

COT_INSTRUCTION_PROMPT = """
You are tasked with creating a detailed step-by-step plan to address a specific problem. Each plan should clearly outline the necessary steps, ensuring clarity and logical flow. Use the following format to present your response:
```
<problem>[The problem description is given here]</problem>
<plan>
   <step id=1>
      [The first step of the plan is described here.]
   <\step>
   <step id=2>
      [The second step of the plan is described here.]
   <\step>
   <step id=3>
      [The third step of the plan is described here.]
   <\step>
   ...
   <step id=[The step number is given here]>
      [The nth step of the plan is described here.]
   <\step>
<\plan>
```

### Instructions:
1. Provide a clear and concise description of the problem.
2. Develop a comprehensive step-by-step plan, ensuring each step is labeled with a unique ID.
3. Maintain logical progression in the steps to facilitate understanding and execution.
"""