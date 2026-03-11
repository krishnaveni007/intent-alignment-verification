"""
LLM-based preference extractor.

Approach TBD. Will implement preference extraction from conversation history
using an LLM. Candidate approaches:
  (1) Staab-style zero-shot prompt extraction — prompt the LLM with the full
      conversation and ask it to enumerate user preferences per dimension.
  (2) DST-style structured slot filling — maintain a belief state over known
      slots and update incrementally at each turn.
  (3) Hybrid — use zero-shot extraction to seed the slot schema, then refine
      via structured updates.

See src/inference/base.py for the interface contract that this class must
satisfy.
"""

from src.inference.base import PreferenceInferrer


class LLMExtractor(PreferenceInferrer):
    """Placeholder LLM-based preference extractor.

    Parameters
    ----------
    model_name : str
        The model identifier to use for inference (e.g. ``'gpt-4o'``).
    api_key : str, optional
        API key for the model provider.  If ``None``, the implementation will
        fall back to the relevant environment variable (e.g.
        ``OPENAI_API_KEY``).
    """

    def __init__(self, model_name: str = "gpt-4o", api_key: str = None) -> None:
        self.model_name = model_name
        self.api_key = api_key

    def infer(self, history: list[dict]) -> dict:
        """Infer preference vector û_t from conversation history.

        Parameters
        ----------
        history : list[dict]
            Conversation turns up to and including turn *t*.

        Returns
        -------
        dict
            û_t mapping dimension -> list of preference strings.
        """
        # TODO: implement LLM-based preference extraction.
        # Choose one of the three candidate approaches described in the module
        # docstring, construct an appropriate prompt from `history`, call the
        # model API, and parse the structured response into the required dict
        # format before returning it.
        return {}
