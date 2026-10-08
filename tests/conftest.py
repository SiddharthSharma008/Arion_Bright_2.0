"""Shared fixtures: a scripted chat model and an offline embedder.

These let the whole graph run end to end without an API key or network,
so the wiring (routing, tool loop, step budget, memory) is tested for real.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Chat model whose reply is computed by a Python function."""

    responder: Callable[[list[BaseMessage], bool], AIMessage]
    tools_bound: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self.model_copy(update={"tools_bound": True})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        reply = self.responder(messages, self.tools_bound)
        return ChatResult(generations=[ChatGeneration(message=reply)])


def system_text(messages: list[BaseMessage]) -> str:
    return next((m.content for m in messages if isinstance(m, SystemMessage)), "")


@pytest.fixture
def use_model(monkeypatch):
    """Patch the graph's LLM factory with a scripted model."""

    def _install(responder):
        from arion.agents import graph as graph_mod

        monkeypatch.setattr(
            graph_mod, "_llm", lambda temperature=0.0: ScriptedChatModel(responder=responder)
        )

    return _install


class HashEmbedder:
    """Deterministic bag-of-words embedder so RAG tests run offline."""

    DIM = 256

    def __call__(self, input):  # chromadb calls with `input`
        vectors = []
        for text in input:
            v = [0.0] * self.DIM
            for word in text.lower().split():
                h = int(hashlib.md5(word.strip(".,:;!?").encode()).hexdigest(), 16)
                v[h % self.DIM] += 1.0
            vectors.append(v)
        return vectors

    def embed_query(self, input):
        return self(input)

    @staticmethod
    def name() -> str:
        return "hash-test-embedder"

    def is_legacy(self) -> bool:
        return False

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config):  # noqa: ARG004
        return HashEmbedder()

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]
