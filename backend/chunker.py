"""
Split transcripts into overlapping, searchable chunks with TF-IDF retrieval.
"""

import math
import re
from dataclasses import dataclass, field
from collections import Counter

from transcriber import TranscriptSegment


@dataclass
class Chunk:
    index: int
    text: str
    start: float  # timestamp in seconds
    end: float
    token_count: int


@dataclass
class ChunkStore:
    """Stores chunks with TF-IDF retrieval."""
    chunks: list[Chunk] = field(default_factory=list)
    _idf: dict[str, float] = field(default_factory=dict)
    _tf: list[dict[str, float]] = field(default_factory=list)

    def build_index(self):
        """Build TF-IDF index from chunks."""
        doc_count = len(self.chunks)
        if doc_count == 0:
            return

        # Document frequency
        df: dict[str, int] = Counter()
        self._tf = []

        for chunk in self.chunks:
            tokens = _tokenize(chunk.text)
            tf = Counter(tokens)
            total = len(tokens) or 1
            self._tf.append({t: c / total for t, c in tf.items()})
            for t in set(tokens):
                df[t] += 1

        self._idf = {
            t: math.log((doc_count + 1) / (freq + 1)) + 1
            for t, freq in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return top-k chunk indices with relevance scores."""
        query_tokens = _tokenize(query)
        if not query_tokens or not self._tf:
            return [(i, 0.0) for i in range(min(top_k, len(self.chunks)))]

        scores = []
        for i, tf in enumerate(self._tf):
            score = sum(
                tf.get(t, 0.0) * self._idf.get(t, 0.0)
                for t in query_tokens
            )
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_context(self, chunk_indices: list[int], max_tokens: int = 2000) -> str:
        """Get combined text from selected chunks, respecting token budget."""
        texts = []
        total = 0
        for idx in sorted(chunk_indices):
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                if total + chunk.token_count > max_tokens:
                    break
                ts = _fmt_time(chunk.start)
                texts.append(f"[{ts}] {chunk.text}")
                total += chunk.token_count
        return "\n".join(texts)


def chunk_transcript(
    segments: list[TranscriptSegment],
    target_tokens: int = 500,
    overlap_tokens: int = 100,
) -> ChunkStore:
    """Split transcript segments into overlapping chunks."""
    if not segments:
        return ChunkStore()

    # Merge all segments into a flat token stream with timestamps
    all_words: list[tuple[str, float, float]] = []
    for seg in segments:
        words = seg.text.split()
        n = len(words) or 1
        duration = seg.end - seg.start
        for i, w in enumerate(words):
            t = seg.start + (i / n) * duration
            all_words.append((w, t, seg.end))

    chunks = []
    idx = 0
    chunk_index = 0

    while idx < len(all_words):
        end_idx = min(idx + target_tokens, len(all_words))
        chunk_words = all_words[idx:end_idx]

        text = " ".join(w for w, _, _ in chunk_words)
        start_ts = chunk_words[0][1]
        end_ts = chunk_words[-1][2]
        token_count = len(chunk_words)

        chunks.append(Chunk(
            index=chunk_index,
            text=text,
            start=start_ts,
            end=end_ts,
            token_count=token_count,
        ))
        chunk_index += 1

        # Advance by (target - overlap) words
        step = max(1, target_tokens - overlap_tokens)
        idx += step

    store = ChunkStore(chunks=chunks)
    store.build_index()
    return store


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization with lowercasing and stopword removal."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "between",
        "through", "during", "before", "after", "and", "but", "or", "nor",
        "not", "so", "yet", "both", "either", "neither", "each", "every",
        "all", "any", "few", "more", "most", "other", "some", "such", "no",
        "only", "own", "same", "than", "too", "very", "just", "because",
        "if", "when", "where", "how", "what", "which", "who", "whom",
        "this", "that", "these", "those", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "it", "its",
        "they", "them", "their",
    }
    return [w for w in words if w not in stopwords and len(w) > 1]


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"
