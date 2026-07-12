"""Progress-bar policy for training entry points.

stable-baselines3's ``model.learn(progress_bar=True)`` renders through a
``ProgressBarCallback`` that hard-requires the optional ``tqdm`` and ``rich``
packages and only makes sense at an interactive terminal. Passing ``True``
unconditionally raises ``ImportError`` on any environment where those extras
are absent (CI, headless servers, minimal installs) and spams non-interactive
logs even when they are present.

``default_progress_bar()`` returns ``True`` only when stdout is a real TTY
*and* both packages are importable, so headless/CI runs silently disable the
bar instead of crashing, while interactive users who installed the optional
``progress`` extra still get it.
"""
import sys
from importlib.util import find_spec


def default_progress_bar() -> bool:
    """Whether to show the sb3 training progress bar in this environment."""
    stdout = getattr(sys, "stdout", None)
    if stdout is None or not hasattr(stdout, "isatty") or not stdout.isatty():
        return False
    return find_spec("tqdm") is not None and find_spec("rich") is not None
