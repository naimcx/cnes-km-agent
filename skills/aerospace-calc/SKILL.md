---
name: aerospace-calc
description: Expert guidelines and numerical calculations for aerospace and engineering data.
---

# Guidelines for Physical and Geometric Calculations in Aerospace

This skill governs the validation and mathematical calculations performed on data extracted from CNES technical documentation.

## Anti-Hallucination Rigor Rules
1. **Strict Prohibition of Mental Arithmetic:**
   - The LLM must NEVER perform divisions, complex products, trigonometry, or volume calculations directly within the text.
   - As soon as a calculation is required on data originating from PDFs, draft the Python script and delegate its execution to the `calculate_aerospace_math` tool.

2. **Pre-Injected Libraries and Constants (No `import` Required):**
   - Use `math` for standard mathematical functions and constants: `math.pi`, `math.e`, `math.sqrt()`, `math.sin()`, etc.
   - Use `constants` (scipy.constants) for physical and aerospace constants: $g_0 =$ `constants.g` ($9.80665 \text{ m/s}^2$).
   - Use `scipy` for advanced scientific computing tools where necessary.

3. **CNES Unit Conventions:**
   - Dimensions extracted from schematics (fairing blueprints, adapters) are in millimeters (mm). Convert to meters (m) to compute volumes in $\text{m}^3$.
   - Express thrust in kN and mass in kg or metric tons.

4. **Expected Execution Format:**
   - Write a concise script that strictly stores the computed output in the final variable `result`.
   - Document every variable with its documentary reference (e.g., `r_base = 5400 / 2000  # Page 5-5`).

5. **Triggering Rule for Calculations:**
   - **Mandatory Precondition:** All numerical variables injected into the code must come from verified extractions.
   - **Formal Prohibition:** No calculation should be executed with assumed dimensions (e.g., `h = 10`, `h = r`, etc.). If a dimension is missing to calculate volume, the calculation must not be executed.
