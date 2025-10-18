"""Analysis service for running statistical analyses.

This service wraps the existing LangGraph workflow and manages analysis execution.
"""

import logging
from datetime import datetime
from io import StringIO

from sqlalchemy.orm import Session

from database.models import Analysis, AnalysisStatus
from statmate.api.services.dataset_service import DatasetService
from statmate.api.services.storage_service import StorageService
from statmate.workflow.statmate_flow import compiled

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for managing statistical analysis execution."""

    @staticmethod
    def create_analysis(
        db: Session,
        dataset_id: str,
        selected_columns: list[str] | None = None,
        configuration: dict | None = None,
        model_name: str | None = None,
        provider: str | None = None,
    ) -> Analysis:
        """Create a new analysis record.

        Args:
            db: Database session
            dataset_id: UUID of dataset to analyze
            selected_columns: Columns to analyze (None = all)
            configuration: Optional analysis configuration
            model_name: AI model to use for analysis
            provider: Model provider (e.g., 'openai', 'ollama')

        Returns:
            Created Analysis model
        """
        analysis = Analysis(
            dataset_id=dataset_id,
            status=AnalysisStatus.PENDING,
            selected_columns=selected_columns,
            configuration=configuration or {},
            model_name=model_name,
            provider=provider,
        )

        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        logger.info(f'Created analysis: {analysis.id} with model: {model_name or "default"}')
        return analysis

    @staticmethod
    def get_analysis(db: Session, analysis_id: str) -> Analysis | None:
        """Get an analysis by ID.

        Args:
            db: Database session
            analysis_id: Analysis UUID

        Returns:
            Analysis model or None if not found
        """
        return db.query(Analysis).filter(Analysis.id == analysis_id).first()

    @staticmethod
    def list_analyses(db: Session, skip: int = 0, limit: int = 100) -> list[Analysis]:
        """List all analyses with pagination.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of Analysis models
        """
        return db.query(Analysis).order_by(Analysis.start_time.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def run_analysis(db: Session, analysis_id: str) -> Analysis:
        """Execute a statistical analysis.

        Args:
            db: Database session
            analysis_id: Analysis UUID

        Returns:
            Updated Analysis model

        Raises:
            ValueError: If analysis or dataset not found
        """
        analysis = AnalysisService.get_analysis(db, analysis_id)
        if not analysis:
            msg = f'Analysis not found: {analysis_id}'
            raise ValueError(msg)

        # Update status to running
        analysis.status = AnalysisStatus.RUNNING
        analysis.start_time = datetime.utcnow()
        db.commit()

        # Set up logging capture
        log_stream = StringIO()
        log_handler = logging.StreamHandler(log_stream)
        log_handler.setLevel(logging.INFO)
        workflow_logger = logging.getLogger('statmate.workflow')
        workflow_logger.addHandler(log_handler)

        try:
            # Load dataset
            df = DatasetService.load_dataset_dataframe(db, analysis.dataset_id)
            if df is None:
                raise ValueError(f'Dataset not found: {analysis.dataset_id}')

            # Filter columns if specified
            if analysis.selected_columns:
                df = df[analysis.selected_columns]

            # Prepare initial state for workflow
            initial_state = {
                'df': df,
                'secondary_df': None,
                'target_columns': [],
                'paired': False,
                'data_type': None,
                'do_association': False,
                'number_of_samples': 0,
                'results': [],
                'probabilities': {},
                'model_name': analysis.model_name,  # Pass selected model
                'provider': analysis.provider,      # Pass selected provider
            }

            # Run the LangGraph workflow
            model_info = f' with {analysis.model_name}' if analysis.model_name else ' with default model'
            logger.info('=' * 80)
            logger.info(f'🚀 STARTING ANALYSIS: {analysis_id}')
            logger.info(f'   Model: {analysis.model_name or "default"}')
            logger.info(f'   Dataset ID: {analysis.dataset_id}')
            logger.info(f'   Rows: {len(df)}, Columns: {len(df.columns)}')
            logger.info('=' * 80)
            messages = []
            probabilities = {}

            for msg, meta in compiled.stream(initial_state, stream_mode='messages'):
                messages.append(str(msg.content))
                # Extract probabilities from state if available
                if hasattr(msg, 'additional_kwargs'):
                    state_probs = msg.additional_kwargs.get('probabilities', {})
                    probabilities.update(state_probs)

            # Combine all messages
            full_output = '\n\n'.join(messages)

            # Extract summary (last message usually contains summary)
            summary = messages[-1] if messages else 'Analysis completed'

            # Extract probabilities from the final state
            if initial_state.get('probabilities'):
                probabilities = initial_state['probabilities']

            # Save results
            results_data = {
                'analysis_id': analysis_id,
                'dataset_id': analysis.dataset_id,
                'selected_columns': analysis.selected_columns,
                'full_output': full_output,
                'messages': messages,
                'probabilities': probabilities,
                'summary': summary,
                'timestamp': datetime.utcnow().isoformat(),
            }

            result_path = StorageService.save_results(analysis_id, results_data)

            # Save log
            log_content = log_stream.getvalue()
            log_path = StorageService.save_log(analysis_id, log_content)

            # Update analysis record
            analysis.status = AnalysisStatus.COMPLETED
            analysis.end_time = datetime.utcnow()
            analysis.result_path = str(result_path)
            analysis.log_path = str(log_path)
            analysis.summary = summary[:1000]  # Truncate if too long
            analysis.probabilities = probabilities

            db.commit()

            logger.info('=' * 80)
            logger.info(f'✅ ANALYSIS COMPLETED: {analysis_id}')
            logger.info(f'   Duration: {(analysis.end_time - analysis.start_time).total_seconds():.2f}s')
            logger.info(f'   Tests performed: {len(probabilities)}')
            logger.info('=' * 80)

        except Exception as e:
            logger.error(f'Analysis failed: {analysis_id} - {e}', exc_info=True)

            # Save error log
            log_content = log_stream.getvalue()
            log_content += f'\n\nERROR: {str(e)}'
            log_path = StorageService.save_log(analysis_id, log_content)

            # Update analysis record
            analysis.status = AnalysisStatus.FAILED
            analysis.end_time = datetime.utcnow()
            analysis.error_message = str(e)
            analysis.log_path = str(log_path)

            db.commit()
            raise

        finally:
            # Remove log handler
            workflow_logger.removeHandler(log_handler)
            log_handler.close()

        return analysis

    @staticmethod
    def get_analysis_results(db: Session, analysis_id: str) -> dict | None:
        """Get detailed analysis results.

        Args:
            db: Database session
            analysis_id: Analysis UUID

        Returns:
            Results dictionary or None if not found
        """
        analysis = AnalysisService.get_analysis(db, analysis_id)
        if not analysis or analysis.status != AnalysisStatus.COMPLETED:
            return None

        # Load results from file
        results_data = StorageService.read_results(analysis_id)
        if not results_data:
            return None

        # Enhance with database metadata
        dataset = DatasetService.get_dataset(db, analysis.dataset_id)
        duration = None
        if analysis.start_time and analysis.end_time:
            duration = (analysis.end_time - analysis.start_time).total_seconds()

        return {
            'id': analysis.id,
            'status': analysis.status.value,
            'dataset_id': analysis.dataset_id,
            'dataset_name': dataset.original_filename if dataset else 'Unknown',
            'model_name': analysis.model_name,
            'provider': analysis.provider,
            'start_time': analysis.start_time,
            'end_time': analysis.end_time,
            'duration_seconds': duration,
            'summary': analysis.summary,
            'probabilities': analysis.probabilities,
            'results_detail': results_data,
            'log_available': bool(analysis.log_path),
        }

    @staticmethod
    def get_analysis_log(db: Session, analysis_id: str) -> str | None:
        """Get analysis execution log.

        Args:
            db: Database session
            analysis_id: Analysis UUID

        Returns:
            Log content or None if not found
        """
        analysis = AnalysisService.get_analysis(db, analysis_id)
        if not analysis or not analysis.log_path:
            return None

        return StorageService.read_log(analysis_id)
