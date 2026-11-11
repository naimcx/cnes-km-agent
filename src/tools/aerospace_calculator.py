import ast
import subprocess
from langchain_core.tools import tool

# TODO: try with gVisor from Google for more security and isolation
# Protect the kernel (using runsc for docker) and limit the attack surface of the container.

MAX_ALLOWED_TIMEOUT_SECONDS = 30
DEFAULT_TIMEOUT_SECONDS = 15

def _ensure_last_value_printed(code_snippet: str) -> str:
    """
    Ensures that if the LLM stores the answer in 'result', 'V', 'volume',
    or leaves a bare expression on the last line (like 'V'), it gets printed.
    """
    try:
        tree = ast.parse(code_snippet)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            # If the last line is a bare expression (e.g. `V`), assign it to `result`
            last_expr = ast.get_source_segment(code_snippet, tree.body[-1])
            if last_expr and not last_expr.strip().startswith("print"):
                lines = code_snippet.splitlines()
                lines[-1] = f"result = {last_expr}"
                code_snippet = "\n".join(lines)
    except Exception:
        pass
    return code_snippet

def execute_in_docker(code_snippet: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    Execute the LLM-generated code in a ephemeral hardened Docker container
    With strict resource limits and no network access.
    """
    normalized_code = _ensure_last_value_printed(code_snippet)

    wrapped_code = (
        "import math\n"
        "try:\n"
        "    import scipy\n"
        "    from scipy import constants\n"
        "except ImportError:\n"
        "    class _Constants:\n"
        "        g = 9.80665\n"
        "        G = 6.67430e-11\n"
        "        c = 299792458.0\n"
        "        pi = math.pi\n"
        "    constants = _Constants()\n\n"
        f"{normalized_code}\n\n"
        "for _var in ('result', 'V', 'volume', 'area', 'ans', 'output'):\n"
        "    if _var in locals():\n"
        "        print(locals()[_var])\n"
        "        break\n"
    )

    # Pass script via 'python -c' to avoid slow Windows-to-WSL2 volume mounts (-v)
    docker_cmd = [
        "docker", "run", "--rm",
        "--network", "none",            # No network access
        "--memory", "128m",             # Limit memory usage
        "--cpus", "0.5",                # Limit CPU usage
        "--pids-limit", "16",           # Limit number of processes
        "--read-only",                  # Filesystem read-only
        "--security-opt", "no-new-privileges:true", # No privilege escalation
        "--user", "1000:1000",          # Run as non-root user
        "python:3.11-slim",             # Use a slim Python image
        "python", "-c", wrapped_code
    ]

    try:
        completed = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=float(timeout)
        )

        if completed.returncode != 0:
            return f"Error : {completed.stderr.strip()}"

        return completed.stdout.strip()

    except subprocess.TimeoutExpired:
        return f"Error: Execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Unexpected error during execution: {str(e)}"


@tool
def calculate_aerospace_math(code: str, estimated_timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    USAGE RULE:
    Do not call this tool for general-purpose code execution. Only use it for aerospace math calculations.
    Do not call this tool until all required parameters have already been explicitly found and validated.
    Never call this tool with guessed, estimated, assumed, or hallucinated values.
    If any required parameter is missing, do not call this tool. Instead, ask the LLM to find the missing parameter first.

    FOR CALCULATIONS:
    Execute a deterministic aerospace math calculation using math and scipy constants in an isolated Docker container.
    The LLM must provide a complete and correct Python code snippet that computes the desired result.
    Store the final result in the variable 'result'.
    """
    clean_code = code.strip().strip("`").replace("python", "").strip()
    timeout = max(1, min(estimated_timeout, MAX_ALLOWED_TIMEOUT_SECONDS))
    return execute_in_docker(clean_code, timeout=timeout)
