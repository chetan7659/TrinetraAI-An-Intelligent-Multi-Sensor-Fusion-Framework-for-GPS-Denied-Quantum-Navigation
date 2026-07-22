"""Trinetra CLI entry point."""

from trinetra.infrastructure.logging import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    """Trinetra CLI entry point."""
    logger.info("Trinetra-AI initialized successfully.")
    print("Trinetra-AI initialized successfully.")


if __name__ == "__main__":
    main()
