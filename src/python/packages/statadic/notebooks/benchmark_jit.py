"""Compatibility launcher for the internal performance application.

Importing this module runs no benchmark and changes no JAX configuration.
Prefer ``just python::performance::smoke`` from the repository root.
The original measurements and script are retained in apps/performance/history.
"""


def main() -> None:
    """Run the new explicitly requested performance command."""
    from statadic_performance.cli import main as performance_main

    performance_main()


if __name__ == "__main__":
    main()
