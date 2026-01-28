import unittest

import pytest

from rag.chunking import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_chunk_text_respects_size_and_overlap(self):
        text = "one two three four five six seven eight nine ten"
        chunks = chunk_text(text, chunk_size=4, overlap=1)
        self.assertEqual(
            chunks,
            [
                "one two three four",
                "four five six seven",
                "seven eight nine ten",
            ],
        )

    def test_chunk_text_empty(self):
        self.assertEqual(chunk_text("", chunk_size=3, overlap=1), [])

    def test_chunk_text_handles_whitespace(self):
        text = "  one   two\nthree\tfour  "
        chunks = chunk_text(text, chunk_size=2, overlap=0)
        self.assertEqual(chunks, ["one two", "three four"])

    def test_chunk_text_single_partial_chunk(self):
        chunks = chunk_text("alpha beta", chunk_size=5, overlap=1)
        self.assertEqual(chunks, ["alpha beta"])


def test_chunk_text_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        chunk_text("alpha", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("alpha", chunk_size=2, overlap=-1)
    with pytest.raises(ValueError):
        chunk_text("alpha beta", chunk_size=2, overlap=2)


if __name__ == "__main__":
    unittest.main()
