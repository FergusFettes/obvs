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
    assert len(embed.embeddings) == 1
    assert embed.embeddings[0].shape[1] == 768


def test_gpt2_multiple_words():
    embed = Embedding([" hello", " world"])
    assert embed.embedding.shape[0] == 768
    assert len(embed.embeddings) == 2
    assert embed.embeddings[0].shape[1] == 768
