"""Serve the local presentation demo. No model is loaded."""

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(
        *args, directory=str(root), **kwargs
    )
    server = ThreadingHTTPServer(("127.0.0.1", 8765), handler)
    print("Demo: http://127.0.0.1:8765", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
