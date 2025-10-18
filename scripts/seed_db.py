#!/usr/bin/env python3
"""Seed the database with sample data for development/testing."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta

from database.models import Analysis, AnalysisStatus, Dataset, ScheduledTask, TaskStatus, TaskType
from database.session import SessionLocal


def seed_database() -> None:
    """Seed database with sample datasets, analyses, and tasks."""
    db = SessionLocal()

    try:
        # Create sample datasets
        dataset1 = Dataset(
            filename='sample_continuous_paired_20250101_120000.parquet',
            original_filename='patient_blood_pressure.csv',
            upload_timestamp=datetime.utcnow() - timedelta(days=5),
            file_size=15360,
            row_count=80,
            column_names=['patient_id', 'before_treatment', 'after_treatment'],
            data_types={'patient_id': 'int64', 'before_treatment': 'float64', 'after_treatment': 'float64'},
            description='Blood pressure measurements before and after treatment',
        )

        dataset2 = Dataset(
            filename='sample_categorical_20250103_150000.parquet',
            original_filename='smoking_exercise_study.csv',
            upload_timestamp=datetime.utcnow() - timedelta(days=3),
            file_size=25600,
            row_count=200,
            column_names=['user_id', 'smoker', 'exercise_level'],
            data_types={'user_id': 'int64', 'smoker': 'object', 'exercise_level': 'object'},
            description='Study on smoking habits and exercise levels',
        )

        db.add_all([dataset1, dataset2])
        db.commit()
        db.refresh(dataset1)
        db.refresh(dataset2)

        print(f'  ✓ Created dataset: {dataset1.original_filename} (ID: {dataset1.id})')
        print(f'  ✓ Created dataset: {dataset2.original_filename} (ID: {dataset2.id})')

        # Create sample completed analysis
        analysis1 = Analysis(
            dataset_id=dataset1.id,
            status=AnalysisStatus.COMPLETED,
            selected_columns=['before_treatment', 'after_treatment'],
            configuration={},
            start_time=datetime.utcnow() - timedelta(hours=2),
            end_time=datetime.utcnow() - timedelta(hours=2, minutes=-5),
            result_path=f'data/results/{dataset1.id}/summary.json',
            log_path=f'data/logs/{dataset1.id}.log',
            summary='Paired t-test showed significant difference (p < 0.001)',
            probabilities={'paired_t_test': 0.0003, 'shapiro_wilk': 0.15},
        )

        # Create sample pending analysis
        analysis2 = Analysis(
            dataset_id=dataset2.id,
            status=AnalysisStatus.PENDING,
            selected_columns=['smoker', 'exercise_level'],
            configuration={},
        )

        db.add_all([analysis1, analysis2])
        db.commit()

        print(f'  ✓ Created analysis: {analysis1.id} (status: {analysis1.status.value})')
        print(f'  ✓ Created analysis: {analysis2.id} (status: {analysis2.status.value})')

        # Create sample scheduled tasks
        task1 = ScheduledTask(
            name='Nightly Regression Analysis',
            task_type=TaskType.RECURRING,
            dataset_id=dataset1.id,
            selected_columns=['before_treatment', 'after_treatment'],
            configuration={},
            schedule='0 2 * * *',  # Every day at 2 AM
            status=TaskStatus.ACTIVE,
            next_run=datetime.utcnow() + timedelta(hours=8),
            run_count=0,
        )

        task2 = ScheduledTask(
            name='Weekly User Cohort Report',
            task_type=TaskType.RECURRING,
            dataset_id=dataset2.id,
            selected_columns=['smoker', 'exercise_level'],
            configuration={},
            schedule='0 9 * * 1',  # Every Monday at 9 AM
            status=TaskStatus.ACTIVE,
            next_run=datetime.utcnow() + timedelta(days=3),
            run_count=5,
            last_run=datetime.utcnow() - timedelta(days=4),
        )

        db.add_all([task1, task2])
        db.commit()

        print(f'  ✓ Created scheduled task: {task1.name} (ID: {task1.id})')
        print(f'  ✓ Created scheduled task: {task2.name} (ID: {task2.id})')

    except Exception as e:
        print(f'  ✗ Error seeding database: {e}')
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == '__main__':
    seed_database()
