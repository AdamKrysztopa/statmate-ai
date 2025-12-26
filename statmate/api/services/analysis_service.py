"""Analysis service for running statistical analyses.

This service wraps the existing LangGraph workflow and manages analysis execution.
"""

import logging
from datetime import datetime
from io import StringIO
from typing import Any

from sqlalchemy.orm import Session

from config.settings import settings
from database.models import Analysis, AnalysisStatus
from statmate.api.services.dataset_service import DatasetService
from statmate.api.services.storage_service import StorageService
from statmate.api.services.visualization_service import VisualizationService
from statmate.workflow.statmate_flow_refactored import StatMateWorkflow

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect provider rate limit errors without tight coupling to SDK types."""
    msg = str(exc).lower()
    return 'rate limit' in msg or 'rate_limit_exceeded' in msg


def _state_value(state: Any, key: str, default: Any) -> Any:
    """Safely fetch a key/attribute from a WorkflowState or dict."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


class AnalysisService:
    """Service for managing statistical analysis execution."""

    @staticmethod
    def create_analysis(
        db: Session,
        dataset_id: str,
        *,
        selected_columns: list[str] | None = None,
        configuration: dict[str, Any] | None = None,
        model_name: str | None = None,
        provider: str | None = None,
        user_id: str | None = None,
    ) -> Analysis:
        """Create a new analysis record.

        Args:
            db: The database session.
            dataset_id: The UUID of the dataset to analyze.
            selected_columns: The columns to analyze. If None, all columns are used.
            configuration: Optional analysis configuration.
            model_name: The AI model to use for the analysis.
            provider: The model provider (e.g., 'openai', 'ollama').

        Returns:
            The created Analysis model.
        """
        # Ownership check when applicable
        from config.settings import settings
        ds = DatasetService.get_dataset(db, dataset_id, user_id=user_id if settings.AUTH_REQUIRED else None)
        if settings.AUTH_REQUIRED and not ds:
            raise ValueError('Dataset not found or not owned by user')

        analysis = Analysis(
            dataset_id=dataset_id,
            user_id=user_id,
            status=AnalysisStatus.PENDING,
            selected_columns=selected_columns,
            configuration=configuration or {},
            model_name=model_name,
            provider=provider,
        )

        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        logger.info('Created analysis: %s with model: %s', analysis.id, model_name or 'default')
        return analysis

    @staticmethod
    def get_analysis(db: Session, analysis_id: str, *, user_id: str | None = None) -> Analysis | None:
        """Get an analysis by its ID.

        Args:
            db: The database session.
            analysis_id: The analysis UUID.

        Returns:
            The Analysis model or None if not found.
        """
        query = db.query(Analysis).filter(Analysis.id == analysis_id)
        if user_id:
            query = query.filter(Analysis.user_id == user_id)
        return query.first()

    @staticmethod
    def list_analyses(db: Session, *, skip: int = 0, limit: int = 100, user_id: str | None = None) -> list[Analysis]:
        """List all analyses with pagination.

        Args:
            db: The database session.
            skip: The number of records to skip.
            limit: The maximum number of records to return.

        Returns:
            A list of Analysis models.
        """
        query = db.query(Analysis)
        if user_id:
            query = query.filter(Analysis.user_id == user_id)
        return query.order_by(Analysis.start_time.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def _update_analysis_status(
        db: Session,
        analysis: Analysis,
        *,
        status: AnalysisStatus,
        error_message: str | None = None,
    ) -> None:
        """Private helper to update the analysis status and commit to the database.

        Args:
            db: The database session.
            analysis: The analysis object to update.
            status: The new status to set.
            error_message: An optional error message if the status is FAILED.
        """
        analysis.status = status
        analysis.end_time = datetime.utcnow()
        if error_message:
            analysis.error_message = error_message
        db.commit()

    @staticmethod
    def run_analysis(db: Session, analysis_id: str, *, user_id: str | None = None) -> Analysis:
        """Execute a statistical analysis using the refactored workflow.

        Args:
            db: The database session.
            analysis_id: The analysis UUID.

        Returns:
            The updated Analysis model.

        Raises:
            ValueError: If the analysis or its associated dataset is not found.
        """
        analysis = AnalysisService.get_analysis(db, analysis_id)
        if not analysis:
            raise ValueError(f'Analysis not found: {analysis_id}')
        if user_id and analysis.user_id and analysis.user_id != user_id:
            raise ValueError('Analysis does not belong to this user')

        analysis.status = AnalysisStatus.RUNNING
        analysis.start_time = datetime.utcnow()
        db.commit()

        log_stream = StringIO()
        log_handler = logging.StreamHandler(log_stream)
        log_handler.setLevel(logging.INFO)

        log_file = settings.get_log_path(analysis_id)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        log_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        workflow_logger = logging.getLogger('statmate')
        workflow_logger.addHandler(log_handler)
        workflow_logger.addHandler(file_handler)

        # Persist log path immediately so clients can poll the live log
        analysis.log_path = str(log_file)
        db.commit()

        try:
            df = DatasetService.load_dataset_dataframe(db, analysis.dataset_id, user_id=user_id or analysis.user_id)
            if df is None:
                raise ValueError(f'Dataset not found for analysis: {analysis.dataset_id}')

            fallback_attempted = False

            def _run_workflow(model_override: str | None = None) -> Any:
                workflow = StatMateWorkflow()
                return workflow.run(
                    data=df,
                    target_columns=analysis.selected_columns,
                    paired=analysis.configuration.get('paired', False),
                    do_association=analysis.configuration.get('do_association', False),
                    model_name=model_override or analysis.model_name,
                    provider=analysis.provider,
                )

            try:
                # First attempt with requested/default model
                result_state = _run_workflow()

            except Exception as e:
                if (
                    _is_rate_limit_error(e)
                    and not fallback_attempted
                    and analysis.provider in (None, '', 'openai')
                    and settings.OPENAI_FALLBACK_ENABLED
                ):
                    fallback_attempted = True
                    fallback_model = settings.OPENAI_FALLBACK_MODEL
                    log_stream.write(f'\nRate limit hit for {analysis.model_name}, retrying with fallback {fallback_model}\n')
                    logger.warning(
                        'Rate limit for %s, retrying analysis %s with fallback model %s',
                        analysis.model_name,
                        analysis_id,
                        fallback_model,
                    )
                    # Switch to fallback model for this run and future retrievals
                    analysis.model_name = fallback_model
                    db.commit()
                    result_state = _run_workflow(model_override=fallback_model)
                else:
                    raise

            logger.info('=' * 80)
            logger.info('🚀 STARTING ANALYSIS: %s', analysis_id)
            logger.info('   Model: %s', analysis.model_name or 'default')
            logger.info('   Dataset ID: %s', analysis.dataset_id)
            logger.info('   Rows: %d, Columns: %d', len(df), len(df.columns))
            logger.info('=' * 80)

            messages = _state_value(result_state, 'results', [])
            probabilities = _state_value(result_state, 'probabilities', {})
            execution_trace = _state_value(result_state, 'execution_trace', [])
            summary = messages[-1].content if messages else 'Analysis completed without a summary.'
            full_output = '\n\n'.join(msg.content for msg in messages)
            plots = VisualizationService.generate_visualizations(
                df,
                selected_columns=analysis.selected_columns,
                limit=6,
            )

            results_data = {
                'analysis_id': analysis_id,
                'dataset_id': analysis.dataset_id,
                'selected_columns': analysis.selected_columns,
                'full_output': full_output,
                'messages': [msg.content for msg in messages],
                'probabilities': probabilities,
                'summary': summary,
                'execution_trace': execution_trace,
                'plots': plots,
                'timestamp': datetime.utcnow().isoformat(),
            }

            analysis.result_path = str(StorageService.save_results(analysis_id, results_data))
            analysis.summary = summary[:1000]
            analysis.probabilities = probabilities
            AnalysisService._update_analysis_status(db, analysis, status=AnalysisStatus.COMPLETED)

            logger.info('=' * 80)
            logger.info('✅ ANALYSIS COMPLETED: %s', analysis_id)
            if analysis.start_time:
                duration = (datetime.utcnow() - analysis.start_time).total_seconds()
                logger.info('   Duration: %.2fs', duration)
            logger.info('   Tests performed: %d', len(probabilities))
            logger.info('=' * 80)

        except Exception as e:
            logger.error('Analysis failed: %s - %s', analysis_id, e, exc_info=True)
            AnalysisService._update_analysis_status(db, analysis, status=AnalysisStatus.FAILED, error_message=str(e))
            log_stream.write(f'\n\nERROR: {e}')
            # We re-raise the exception to ensure the caller knows the operation failed.
            raise
        finally:
            log_content = log_stream.getvalue()
            analysis.log_path = str(StorageService.save_log(analysis_id, log_content))
            db.commit()
            workflow_logger.removeHandler(log_handler)
            workflow_logger.removeHandler(file_handler)
            log_handler.close()
            file_handler.close()

        return analysis

    @staticmethod
    def get_analysis_results(db: Session, analysis_id: str, *, user_id: str | None = None) -> dict[str, Any] | None:
        """Get detailed analysis results.

        Args:
            db: The database session.
            analysis_id: The analysis UUID.

        Returns:
            A dictionary with the results or None if not found.
        """
        analysis = AnalysisService.get_analysis(db, analysis_id, user_id=user_id)
        if not analysis or analysis.status != AnalysisStatus.COMPLETED:
            return None

        results_data = StorageService.read_results(analysis_id)
        if not results_data:
            return None

        dataset = DatasetService.get_dataset(db, str(analysis.dataset_id))
        duration = None
        if analysis.start_time and analysis.end_time:
            duration = (analysis.end_time - analysis.start_time).total_seconds()

        return {
            'id': analysis.id,
            'status': analysis.status.value,
            'dataset_id': analysis.dataset_id,
            'dataset_name': dataset.original_filename if dataset else 'Unknown',
            'user_id': analysis.user_id,
            'model_name': analysis.model_name,
            'provider': analysis.provider,
            'start_time': analysis.start_time,
            'end_time': analysis.end_time,
            'duration_seconds': duration,
            'summary': analysis.summary,
            'probabilities': analysis.probabilities,
            'results_detail': results_data,
            'execution_trace': results_data.get('execution_trace'),
            'plots': results_data.get('plots'),
            'log_available': bool(analysis.log_path),
        }

    @staticmethod
    def get_analysis_log(db: Session, analysis_id: str, *, user_id: str | None = None) -> str | None:
        """Get the analysis execution log.

        Args:
            db: The database session.
            analysis_id: The analysis UUID.

        Returns:
            The log content as a string or None if not found.
        """
        analysis = AnalysisService.get_analysis(db, analysis_id, user_id=user_id)
        if not analysis or not analysis.log_path:
            return None

        return StorageService.read_log(analysis_id)
