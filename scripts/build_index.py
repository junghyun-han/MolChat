"""Build and persist the FAISS retrieval index from the bundled corpus.

    python scripts/build_index.py --embedder fingerprint --out data/index

The agent builds the index in memory on demand, so this is only needed if you
want a persisted index (e.g. for serving in P4).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molchat.embeddings import get_embedder  # noqa: E402
from molchat.rag import RagIndex  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default="fingerprint")
    parser.add_argument("--out", default="data/index")
    args = parser.parse_args()

    rag = RagIndex(embedder=get_embedder(args.embedder)).build()
    assert rag.store is not None
    rag.store.save(args.out)
    print(
        f"Built index: {len(rag.store)} molecules | embedder={rag.embedder.name} "
        f"| dim={rag.embedder.dim} -> {args.out}"
    )


if __name__ == "__main__":
    main()
