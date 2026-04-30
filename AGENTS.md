# Coding conventions

* Package management in this project is done with uv.
* Use red/green test-driven development on this project.
* This project relies on asyncio, so all pytest runs need to employ timeouts to prevent hangs.
* Always log exceptions, pass through, or re-raise -- never swallow exceptions with bare `except Exception: pass`.
* Use idiomatic Python with proper type hinting.
* Comment every block of code with a description of its purpose so that humans can tell what's going on.

# Process instructions

* If you are solving a GitHub issue, post the requirements and implementation plan as a comment on the issue before implementing.
* Check @docs/ before your work is complete and make sure the documentation is updated.
