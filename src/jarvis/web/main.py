"""Entry point for Jarvis Web UI."""

import uvicorn


def main() -> None:
    uvicorn.run("jarvis.web.app:app", host="0.0.0.0", port=7777, reload=False)


if __name__ == "__main__":
    main()
