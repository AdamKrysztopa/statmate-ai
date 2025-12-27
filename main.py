"""Entry point for StatmateAI."""

import logging

import uvicorn

from config.settings import settings


def main() -> None:
    """Run the FastAPI backend with sensible defaults."""
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    uvicorn.run(
        'statmate.api.main:app',
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level='debug' if settings.DEBUG else 'info',
    )


if __name__ == '__main__':
    main()
