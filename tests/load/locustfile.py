"""Locust stub for StatMate load testing.

Add authentication and dataset seeding before running at scale.
"""

from locust import HttpUser, TaskSet, between, task


class AnalysisTasks(TaskSet):
    dataset_id = ''
    analysis_id = ''

    @task(2)
    def health(self):
        self.client.get('/health')

    @task(1)
    def poll_status(self):
        if self.analysis_id:
            self.client.get(f'/api/v1/analysis/{self.analysis_id}')

    @task(1)
    def stream_placeholder(self):
        # SSE endpoints are not fully supported in Locust without a plugin;
        # keep this as a placeholder so scenarios can be extended.
        pass


class StatmateUser(HttpUser):
    tasks = [AnalysisTasks]
    wait_time = between(1, 3)
