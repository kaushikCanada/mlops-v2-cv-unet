import pyodbc
from datetime import datetime
from typing import Optional, Dict, Any
import random
from tenacity import retry
from tenacity.retry import *
from tenacity.stop import *
from tenacity.wait import *


class MLPipelineLogger:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    @retry(
    wait=wait_random(min=5, max=10),  # Wait between 5-10 seconds randomly
    stop=stop_after_attempt(7)        # Stop after 7 attempts total
    )
    def _execute_query(self, query: str, params: tuple):
        with pyodbc.connect(self.connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
    
    def start_pipeline_run(self, run_id: str, pipeline_name: str, user_id: str = None):
        query = """
            INSERT INTO pipeline_runs (run_id, pipeline_name, status, started_at, user_id)
            VALUES (?, ?, 'running', GETUTCDATE(), ?)
        """
        self._execute_query(query, (run_id, pipeline_name, user_id))
    
    def complete_pipeline_run(self, run_id: str, status: str = 'completed'):
        query = """
            UPDATE pipeline_runs 
            SET status = ?, completed_at = GETUTCDATE()
            WHERE run_id = ?
        """
        self._execute_query(query, (status, run_id))
    
    def start_phase(self, run_id: str, phase_name: str):
        query = """
            INSERT INTO pipeline_phases (run_id, phase_name, status, started_at)
            VALUES (?, ?, 'running', GETUTCDATE())
        """
        self._execute_query(query, (run_id, phase_name))
    
    def complete_phase(self, run_id: str, phase_name: str, error: str = None):
        status = 'failed' if error else 'completed'
        query = """
            UPDATE pipeline_phases 
            SET status = ?, completed_at = GETUTCDATE(), error_message = ?
            WHERE run_id = ? AND phase_name = ?
        """
        self._execute_query(query, (status, error, run_id, phase_name))
    
    def log_metric(self, run_id: str, phase_name: str, metric_name: str, 
                   metric_value: float, epoch: int = None, step: int = None):
        query = """
            INSERT INTO phase_metrics (run_id, phase_name, metric_name, metric_value, epoch, step)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        print('yes')
        self._execute_query(query, (run_id, phase_name, metric_name, metric_value, epoch, step))
    
    def log_epoch_metrics(self, run_id: str, phase_name: str, metrics: Dict[str, float], epoch: int):
        for metric_name, metric_value in metrics.items():
            self.log_metric(run_id, phase_name, metric_name, metric_value, epoch=epoch)

    # Query methods for web display
    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        query = """
            SELECT r.run_id, r.pipeline_name, r.status, r.created_at, r.user_id,
                   p.phase_name, p.status as phase_status, p.started_at, p.completed_at
            FROM pipeline_runs r
            LEFT JOIN pipeline_phases p ON r.run_id = p.run_id
            WHERE r.run_id = ?
            ORDER BY p.started_at
        """
        with pyodbc.connect(self.connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (run_id,))
            rows = cursor.fetchall()
            
            if not rows:
                return None
            
            result = {
                'run_id': rows[0].run_id,
                'pipeline_name': rows[0].pipeline_name,
                'status': rows[0].status,
                'created_at': rows[0].created_at,
                'user_id': rows[0].user_id,
                'phases': []
            }
            
            for row in rows:
                if row.phase_name:
                    result['phases'].append({
                        'phase_name': row.phase_name,
                        'status': row.phase_status,
                        'started_at': row.started_at,
                        'completed_at': row.completed_at
                    })
            
            return result
    
    def get_phase_metrics(self, run_id: str, phase_name: str) -> list:
        query = """
            SELECT metric_name, metric_value, epoch, step, logged_at
            FROM phase_metrics
            WHERE run_id = ? AND phase_name = ?
            ORDER BY epoch, step, logged_at
        """
        with pyodbc.connect(self.connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (run_id, phase_name))
            return [dict(zip([column[0] for column in cursor.description], row)) 
                   for row in cursor.fetchall()]
