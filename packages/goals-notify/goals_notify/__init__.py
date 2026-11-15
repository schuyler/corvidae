"""Goals notify: a goal's state changes go to the channel that created it (#841).

Red-phase scaffold: the class exists so the tests can import it.
"""

from __future__ import annotations

from corvidae.hooks import CorvidaePlugin


class GoalsNotifyPlugin(CorvidaePlugin):
    """Sends each goal state change to the goal's origin channel."""
