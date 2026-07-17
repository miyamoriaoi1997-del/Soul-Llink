from __future__ import annotations

from pcltm.transcript_chunker import chunk_transcript


def test_chunks_round_trip_exact_source_spans() -> None:
    text = "第一段说明。\r\n\r\n```python\r\nprint('x')\r\n```\r\n\r\n最后一段🙂"

    chunks = chunk_transcript(text, max_chars=24, overlap_chars=6)

    assert len(chunks) >= 3
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert text[chunk.start_char:chunk.end_char] == chunk.text
        assert chunk.start_char < chunk.end_char
        assert chunk.sha256
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text)


def test_long_unbroken_text_is_chunked_without_character_loss() -> None:
    text = "永久证据" * 40

    chunks = chunk_transcript(text, max_chars=30, overlap_chars=5)

    assert all(len(chunk.text) <= 30 for chunk in chunks)
    covered = [False] * len(text)
    for chunk in chunks:
        for index in range(chunk.start_char, chunk.end_char):
            covered[index] = True
    assert all(covered)
