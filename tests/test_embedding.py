from __future__ import annotations

from obvs.utils import Embedding


def test_gpt2_centroid():
    embedding = Embedding().embedding
    assert embedding.shape[0] == 768


def test_gpt2_single_word():
    embedding = Embedding(" the").embedding
    assert embedding.shape[0] == 768


def test_gpt2_long_input():
    embed = Embedding(" the quick brown fox jumps over the")
    assert embed.embedding.shape[0] == 768


def test_gpt2_multiple_words():
    embed = Embedding([" hello", " world"])
    assert embed.embedding.shape[0] == 768


def test_gpt2_multiple_sentences():
    embed = Embedding([" hello there", " what a world"])
    assert embed.embedding.shape[0] == 768


def test_gpt2_averaging():
    embed1 = Embedding(" cat")
    embed2 = Embedding(" cat cat cat")

    assert embed1.embedding.allclose(embed2.embedding)

    embed3 = Embedding([" cat cat cat", " cat cat"])

    assert embed1.embedding.allclose(embed3.embedding)
