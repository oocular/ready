"""Nox config."""

import pathlib

import nox
import nox_uv

# Options to modify nox behaviour
nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = [
    "lint",
    "tests",
]

ALL_PYTHON = [
    "3.13",
    "3.14",
]


@nox_uv.session(
    uv_no_install_project=True,
    uv_only_groups=["build"],
)
def build(session: nox.Session) -> None:
    """Build an SDist and wheel."""
    session.run("python", "-m", "build")


@nox_uv.session(uv_groups=["docs"])
def docs(session: nox.Session) -> None:
    """Build the docs. Pass "serve" to serve."""
    session.run("zensical", "build", "--strict")

    if session.posargs:
        if "serve" in session.posargs:
            session.run("zensical", "serve", "--strict")
        else:
            session.log("Unsupported argument to docs")
