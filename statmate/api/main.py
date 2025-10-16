"""FastAPI application for StatmateAI.

This is the main application entry point that configures and runs the API server.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from database.session import init_db
from statmate.api.routes import analysis, datasets, results, tasks
from statmate.api.scheduler import init_scheduler, shutdown_scheduler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info('Starting StatmateAI API server...')

    # Initialize database
    try:
        init_db()
        logger.info('Database initialized')
    except Exception as e:
        logger.error(f'Failed to initialize database: {e}', exc_info=True)
        raise

    # Initialize model factory with settings
    try:
        from statmate.workflow.model_factory import initialize_default_factory

        multi_model_config = settings.create_multi_model_config()
        initialize_default_factory(multi_model_config)
        logger.info('Model factory initialized')

        # Log available models
        from statmate.workflow.model_factory import get_default_factory

        factory = get_default_factory()
        available_models = factory.list_available_models(for_tools=True)
        if available_models:
            model_names = [m.display_name for m in available_models]
            logger.info(f'Available reasoning models: {", ".join(model_names)}')
        else:
            logger.warning('No reasoning models configured! Please set API keys in .env file.')
    except Exception as e:
        logger.error(f'Failed to initialize model factory: {e}', exc_info=True)
        # Don't raise - will use default config as fallback

    # Initialize scheduler
    try:
        init_scheduler()
        logger.info('Scheduler initialized')
    except Exception as e:
        logger.error(f'Failed to initialize scheduler: {e}', exc_info=True)
        # Don't raise - scheduler is optional

    logger.info(f'StatmateAI API server started on {settings.API_HOST}:{settings.API_PORT}')
    logger.info(f'API documentation available at http://{settings.API_HOST}:{settings.API_PORT}/docs')

    yield

    # Shutdown
    logger.info('Shutting down StatmateAI API server...')
    shutdown_scheduler()
    logger.info('Scheduler shut down')
    logger.info('StatmateAI API server stopped')


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description='AI-driven statistical analysis tool for clinical and observational research',
    lifespan=lifespan,
    docs_url='/docs',
    redoc_url='/redoc',
    openapi_url=f'{settings.API_PREFIX}/openapi.json',
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# Health check endpoint
@app.get('/health')
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Dictionary with health status
    """
    return {
        'status': 'healthy',
        'version': settings.APP_VERSION,
        'environment': settings.ENVIRONMENT,
    }


@app.get('/')
async def root() -> dict[str, str]:
    """Root endpoint with API information.

    Returns:
        Dictionary with API information
    """
    return {
        'name': settings.APP_NAME,
        'version': settings.APP_VERSION,
        'docs': f'{settings.API_PREFIX}/docs',
        'health': '/health',
    }


# Include routers
app.include_router(datasets.router, prefix=settings.API_PREFIX)
app.include_router(analysis.router, prefix=settings.API_PREFIX)
app.include_router(tasks.router, prefix=settings.API_PREFIX)
app.include_router(results.router, prefix=settings.API_PREFIX)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions.

    Args:
        request: Request object
        exc: Exception

    Returns:
        JSON error response
    """
    logger.error(f'Unhandled exception: {exc}', exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            'detail': 'Internal server error',
            'error': str(exc) if settings.DEBUG else 'An unexpected error occurred',
        },
    )


if __name__ == '__main__':
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Run server
    uvicorn.run(
        'statmate.api.main:app',
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level='debug' if settings.DEBUG else 'info',
    )
