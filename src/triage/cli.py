"""The `triage` command."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

import uvicorn

from triage import db
from triage.api import create_app
from triage.dataset import checks, load_corpus, load_labelled
from triage.dataset.generate import write_files
from triage.embeddings import SNAPSHOT_FILE, LocalEmbedder, Snapshot
from triage.mcp_server import serve_stdio
from triage.retrieval import incident_query
from triage.runtime import build_embedder, open_retriever
from triage.settings import Settings


def _data_generate(settings: Settings, args: argparse.Namespace) -> int:
    write_files(settings.data_dir)
    print(f"wrote the corpus, the incidents and the labels to {settings.data_dir}/")
    return 0


def _data_check(settings: Settings, args: argparse.Namespace) -> int:
    problems = checks.run_all(settings.data_dir)
    for problem in problems:
        print(f"problem: {problem}")
    if problems:
        return 1
    print(json.dumps(checks.summary(settings.data_dir), indent=2))
    return 0


def _data_embed(settings: Settings, args: argparse.Namespace) -> int:
    """Brings the snapshot in line with the data.

    A vector that is already in the snapshot is kept, so that running this on another machine
    does not move every vector by a rounding error. `--rebuild` computes all of them again.
    """
    path = settings.data_dir / SNAPSHOT_FILE
    old = Snapshot() if args.rebuild or not path.exists() else Snapshot.load(path)
    corpus = load_corpus(settings.data_dir)
    labelled = load_labelled(settings.data_dir)
    texts = [document.text for document in corpus]
    texts += [incident_query(item.incident) for item in labelled]
    missing = [text for text in texts if old.get(text) is None]
    computed = LocalEmbedder(settings.embedding_cache_dir).embed(missing) if missing else []
    fresh = dict(zip(missing, computed, strict=True))
    new = Snapshot()
    for text in texts:
        kept = old.get(text)
        new.put(text, kept if kept is not None else fresh[text])
    new.save(path)
    print(f"{path}: {len(texts)} vectors, {len(missing)} computed")
    return 0


def _db_load(settings: Settings, args: argparse.Namespace) -> int:
    corpus = load_corpus(settings.data_dir)
    embedder = build_embedder(settings, snapshot_only=False)
    count = db.load_corpus(settings.database_url, corpus, embedder)
    print(f"loaded {count} documents")
    return 0


def _search(settings: Settings, args: argparse.Namespace) -> int:
    async def run() -> None:
        async with open_retriever(settings, snapshot_only=False) as retriever:
            for hit in await retriever.search(args.query, args.mode, args.limit):
                print(f"{hit.rank}. {hit.id}  {hit.score:.4f}  {hit.title}")

    asyncio.run(run())
    return 0


def _serve(settings: Settings, args: argparse.Namespace) -> int:
    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


def _mcp(settings: Settings, args: argparse.Namespace) -> int:
    asyncio.run(serve_stdio(settings))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triage", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="the synthetic data set").add_subparsers(
        dest="data_command", required=True
    )
    data.add_parser("generate", help="render the catalogue into the data files").set_defaults(
        run=_data_generate
    )
    data.add_parser("check", help="check the data files and print their shape").set_defaults(
        run=_data_check
    )
    embed = data.add_parser("embed", help="bring the embedding snapshot in line with the data")
    embed.add_argument("--rebuild", action="store_true", help="compute every vector again")
    embed.set_defaults(run=_data_embed)

    database = commands.add_parser("db", help="the database").add_subparsers(
        dest="db_command", required=True
    )
    database.add_parser("load", help="create the schema and load the corpus").set_defaults(
        run=_db_load
    )

    search = commands.add_parser("search", help="search the corpus")
    search.add_argument("query")
    search.add_argument("--mode", choices=["vector", "fulltext", "hybrid"], default="hybrid")
    search.add_argument("--limit", type=int, default=5)
    search.set_defaults(run=_search)

    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(run=_serve)

    commands.add_parser("mcp", help="serve the evidence tools over stdio").set_defaults(run=_mcp)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(args.run(Settings(), args))


if __name__ == "__main__":
    main()
