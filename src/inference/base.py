"""
Abstract base class for preference inference from conversation history.

All concrete inferrers must implement :meth:`PreferenceInferrer.infer`, which
takes a conversation history up to turn *t* and returns an inferred preference
vector û_t.  The concrete method :meth:`PreferenceInferrer.infer_incremental`
calls ``infer`` at every turn and collects the full trajectory — used by the
evaluation layer to compute ``misalignment_drift``.
"""

from abc import ABC, abstractmethod


class PreferenceInferrer(ABC):
    """Abstract base class for all preference inference strategies.

    Subclasses must implement :meth:`infer`.  The incremental wrapper
    :meth:`infer_incremental` is provided for free and should not normally
    need to be overridden.
    """

    @abstractmethod
    def infer(self, history: list[dict]) -> dict:
        """Infer a preference vector û_t from conversation history up to turn t.

        Parameters
        ----------
        history : list[dict]
            Conversation turns up to and including turn *t*.  Each turn is a
            dict with at least the keys ``role`` (``'user'`` or
            ``'assistant'``) and ``content`` (str).

        Returns
        -------
        dict
            û_t — a mapping from dimension name (str) to a list of inferred
            preference strings (list[str]).

            Example::

                {
                    "rental_car": ["prefer liability insurance", "prefer child seat"],
                    "apartment": ["prefer rating 7-8"],
                }

            An empty dict is a valid return value when no preferences can be
            inferred from the history provided.
        """

    def infer_incremental(self, history: list[dict]) -> list[dict]:
        """Run :meth:`infer` at every turn and return the full û_t trajectory.

        For a conversation of length *T*, this calls ``infer`` *T* times with
        prefixes of increasing length (turns 1, 2, …, T) and collects the
        returned û_t snapshots.  The resulting list is used by
        :func:`src.evaluation.metrics.misalignment_drift` to measure how
        alignment evolves over the course of a conversation (Hypothesis 2).

        Parameters
        ----------
        history : list[dict]
            Full conversation history as returned by the UserBench loader.

        Returns
        -------
        list[dict]
            A list of *T* û_t snapshots (one per turn), each in the same
            format as the return value of :meth:`infer`.
        """
        snapshots = []
        for t in range(1, len(history) + 1):
            u_hat_t = self.infer(history[:t])
            snapshots.append(u_hat_t)
        return snapshots
