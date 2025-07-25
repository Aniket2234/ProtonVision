"""
Database Management Module for SafeData Pipeline
Handles local SQLite database operations for application state and logs
"""

import sqlite3
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import pickle
import os

class DatabaseManager:
    """Manages local SQLite database for application state and audit logs"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Set default database path
        if db_path is None:
            self.db_path = Path.home() / ".safedata" / "safedata.db"
        else:
            self.db_path = Path(db_path)
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.connection = None
        
    def initialize_database(self):
        """Initialize database with required tables"""
        try:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            
            self._create_tables()
            self.logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def _create_tables(self):
        """Create all required tables"""
        
        # Analysis sessions table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dataset_name TEXT,
                dataset_hash TEXT,
                dataset_rows INTEGER,
                dataset_columns INTEGER,
                status TEXT DEFAULT 'created',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Risk assessments table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS risk_assessments (
                assessment_id TEXT PRIMARY KEY,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                global_risk REAL,
                prosecutor_risk REAL,
                journalist_risk REAL,
                k_anonymity INTEGER,
                l_diversity INTEGER,
                quasi_identifiers TEXT,
                configuration TEXT,
                results TEXT,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
            )
        """)
        
        # Privacy enhancements table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS privacy_enhancements (
                enhancement_id TEXT PRIMARY KEY,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                technique_type TEXT,
                parameters TEXT,
                privacy_budget_used REAL,
                enhancement_log TEXT,
                results TEXT,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
            )
        """)
        
        # Utility measurements table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS utility_measurements (
                measurement_id TEXT PRIMARY KEY,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                overall_utility REAL,
                mae REAL,
                rmse REAL,
                correlation REAL,
                query_accuracy REAL,
                model_performance REAL,
                configuration TEXT,
                results TEXT,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
            )
        """)
        
        # Attack simulations table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS attack_simulations (
                simulation_id TEXT PRIMARY KEY,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                attack_type TEXT,
                success_rate REAL,
                configuration TEXT,
                results TEXT,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
            )
        """)
        
        # Application logs table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS application_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                level TEXT,
                module TEXT,
                message TEXT,
                session_id TEXT,
                user_action TEXT
            )
        """)
        
        # User preferences table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                preference_key TEXT PRIMARY KEY,
                preference_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Dataset metadata table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS dataset_metadata (
                dataset_id TEXT PRIMARY KEY,
                file_path TEXT,
                file_hash TEXT,
                file_size INTEGER,
                rows INTEGER,
                columns INTEGER,
                column_info TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Reports table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS generated_reports (
                report_id TEXT PRIMARY KEY,
                session_id TEXT,
                report_type TEXT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
            )
        """)
        
        self.connection.commit()
        self.logger.debug("Database tables created/verified")
    
    def create_analysis_session(self, dataset_name: str, dataset_hash: str,
                              rows: int, columns: int) -> str:
        """Create a new analysis session"""
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_hash[:8]}"
        
        try:
            self.connection.execute("""
                INSERT INTO analysis_sessions 
                (session_id, dataset_name, dataset_hash, dataset_rows, dataset_columns)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, dataset_name, dataset_hash, rows, columns))
            
            self.connection.commit()
            self.logger.info(f"Created analysis session: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis session: {str(e)}")
            raise
    
    def update_session_status(self, session_id: str, status: str):
        """Update analysis session status"""
        
        try:
            self.connection.execute("""
                UPDATE analysis_sessions 
                SET status = ?, last_updated = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (status, session_id))
            
            self.connection.commit()
            self.logger.debug(f"Updated session {session_id} status to {status}")
            
        except Exception as e:
            self.logger.error(f"Failed to update session status: {str(e)}")
    
    def save_risk_assessment(self, session_id: str, assessment_results: Dict[str, Any],
                           configuration: Dict[str, Any]) -> str:
        """Save risk assessment results"""
        
        assessment_id = f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Extract key metrics
            global_risk = assessment_results.get('global_risk', 0.0)
            prosecutor_risk = assessment_results.get('prosecutor_risk', 0.0)
            journalist_risk = assessment_results.get('journalist_risk', 0.0)
            
            anonymity_metrics = assessment_results.get('anonymity_metrics', {})
            k_anonymity = anonymity_metrics.get('k_anonymity', 0)
            l_diversity = anonymity_metrics.get('l_diversity', 0)
            
            quasi_identifiers = json.dumps(assessment_results.get('quasi_identifiers', []))
            configuration_json = json.dumps(configuration)
            results_json = json.dumps(assessment_results)
            
            self.connection.execute("""
                INSERT INTO risk_assessments 
                (assessment_id, session_id, global_risk, prosecutor_risk, journalist_risk,
                 k_anonymity, l_diversity, quasi_identifiers, configuration, results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (assessment_id, session_id, global_risk, prosecutor_risk, journalist_risk,
                  k_anonymity, l_diversity, quasi_identifiers, configuration_json, results_json))
            
            self.connection.commit()
            self.logger.info(f"Saved risk assessment: {assessment_id}")
            return assessment_id
            
        except Exception as e:
            self.logger.error(f"Failed to save risk assessment: {str(e)}")
            raise
    
    def save_privacy_enhancement(self, session_id: str, enhancement_results: Dict[str, Any],
                               technique_type: str, parameters: Dict[str, Any]) -> str:
        """Save privacy enhancement results"""
        
        enhancement_id = f"enhancement_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            privacy_budget_used = enhancement_results.get('budget_used', 0.0)
            enhancement_log = json.dumps(enhancement_results.get('enhancement_log', []))
            parameters_json = json.dumps(parameters)
            results_json = json.dumps(enhancement_results)
            
            self.connection.execute("""
                INSERT INTO privacy_enhancements 
                (enhancement_id, session_id, technique_type, parameters, 
                 privacy_budget_used, enhancement_log, results)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (enhancement_id, session_id, technique_type, parameters_json,
                  privacy_budget_used, enhancement_log, results_json))
            
            self.connection.commit()
            self.logger.info(f"Saved privacy enhancement: {enhancement_id}")
            return enhancement_id
            
        except Exception as e:
            self.logger.error(f"Failed to save privacy enhancement: {str(e)}")
            raise
    
    def save_utility_measurement(self, session_id: str, utility_results: Dict[str, Any],
                               configuration: Dict[str, Any]) -> str:
        """Save utility measurement results"""
        
        measurement_id = f"utility_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Extract key metrics
            overall_utility = utility_results.get('overall_utility', 0.0)
            mae = utility_results.get('mae', 0.0)
            rmse = utility_results.get('rmse', 0.0)
            correlation = utility_results.get('correlation', 0.0)
            query_accuracy = utility_results.get('query_accuracy', 0.0)
            model_performance = utility_results.get('model_performance', 0.0)
            
            configuration_json = json.dumps(configuration)
            results_json = json.dumps(utility_results)
            
            self.connection.execute("""
                INSERT INTO utility_measurements 
                (measurement_id, session_id, overall_utility, mae, rmse, correlation,
                 query_accuracy, model_performance, configuration, results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (measurement_id, session_id, overall_utility, mae, rmse, correlation,
                  query_accuracy, model_performance, configuration_json, results_json))
            
            self.connection.commit()
            self.logger.info(f"Saved utility measurement: {measurement_id}")
            return measurement_id
            
        except Exception as e:
            self.logger.error(f"Failed to save utility measurement: {str(e)}")
            raise
    
    def save_attack_simulation(self, session_id: str, simulation_results: Dict[str, Any],
                             attack_type: str, configuration: Dict[str, Any]) -> str:
        """Save attack simulation results"""
        
        simulation_id = f"attack_{attack_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            success_rate = simulation_results.get('success_rate', 0.0)
            configuration_json = json.dumps(configuration)
            results_json = json.dumps(simulation_results)
            
            self.connection.execute("""
                INSERT INTO attack_simulations 
                (simulation_id, session_id, attack_type, success_rate, configuration, results)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (simulation_id, session_id, attack_type, success_rate, 
                  configuration_json, results_json))
            
            self.connection.commit()
            self.logger.info(f"Saved attack simulation: {simulation_id}")
            return simulation_id
            
        except Exception as e:
            self.logger.error(f"Failed to save attack simulation: {str(e)}")
            raise
    
    def save_dataset_metadata(self, dataset_id: str, file_path: str, file_hash: str,
                            file_size: int, rows: int, columns: int, 
                            column_info: Dict[str, Any]) -> str:
        """Save dataset metadata"""
        
        try:
            column_info_json = json.dumps(column_info)
            
            self.connection.execute("""
                INSERT OR REPLACE INTO dataset_metadata 
                (dataset_id, file_path, file_hash, file_size, rows, columns, column_info)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (dataset_id, file_path, file_hash, file_size, rows, columns, column_info_json))
            
            self.connection.commit()
            self.logger.info(f"Saved dataset metadata: {dataset_id}")
            return dataset_id
            
        except Exception as e:
            self.logger.error(f"Failed to save dataset metadata: {str(e)}")
            raise
    
    def save_generated_report(self, session_id: str, report_type: str, file_path: str) -> str:
        """Save generated report information"""
        
        report_id = f"report_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.connection.execute("""
                INSERT INTO generated_reports 
                (report_id, session_id, report_type, file_path)
                VALUES (?, ?, ?, ?)
            """, (report_id, session_id, report_type, file_path))
            
            self.connection.commit()
            self.logger.info(f"Saved report metadata: {report_id}")
            return report_id
            
        except Exception as e:
            self.logger.error(f"Failed to save report metadata: {str(e)}")
            raise
    
    def log_application_event(self, level: str, module: str, message: str,
                            session_id: Optional[str] = None, user_action: Optional[str] = None):
        """Log application events"""
        
        try:
            self.connection.execute("""
                INSERT INTO application_logs 
                (level, module, message, session_id, user_action)
                VALUES (?, ?, ?, ?, ?)
            """, (level, module, message, session_id, user_action))
            
            self.connection.commit()
            
        except Exception as e:
            # Don't log errors for logging failures to avoid recursion
            print(f"Failed to log application event: {str(e)}")
    
    def get_analysis_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent analysis sessions"""
        
        try:
            cursor = self.connection.execute("""
                SELECT * FROM analysis_sessions 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append(dict(row))
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"Failed to get analysis sessions: {str(e)}")
            return []
    
    def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """Get all results for a specific session"""
        
        try:
            results = {
                'session_info': None,
                'risk_assessments': [],
                'privacy_enhancements': [],
                'utility_measurements': [],
                'attack_simulations': [],
                'reports': []
            }
            
            # Get session info
            cursor = self.connection.execute("""
                SELECT * FROM analysis_sessions WHERE session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
            if row:
                results['session_info'] = dict(row)
            
            # Get risk assessments
            cursor = self.connection.execute("""
                SELECT * FROM risk_assessments WHERE session_id = ?
                ORDER BY created_at DESC
            """, (session_id,))
            for row in cursor.fetchall():
                assessment = dict(row)
                # Parse JSON fields
                assessment['quasi_identifiers'] = json.loads(assessment['quasi_identifiers'])
                assessment['configuration'] = json.loads(assessment['configuration'])
                assessment['results'] = json.loads(assessment['results'])
                results['risk_assessments'].append(assessment)
            
            # Get privacy enhancements
            cursor = self.connection.execute("""
                SELECT * FROM privacy_enhancements WHERE session_id = ?
                ORDER BY created_at DESC
            """, (session_id,))
            for row in cursor.fetchall():
                enhancement = dict(row)
                enhancement['parameters'] = json.loads(enhancement['parameters'])
                enhancement['enhancement_log'] = json.loads(enhancement['enhancement_log'])
                enhancement['results'] = json.loads(enhancement['results'])
                results['privacy_enhancements'].append(enhancement)
            
            # Get utility measurements
            cursor = self.connection.execute("""
                SELECT * FROM utility_measurements WHERE session_id = ?
                ORDER BY created_at DESC
            """, (session_id,))
            for row in cursor.fetchall():
                measurement = dict(row)
                measurement['configuration'] = json.loads(measurement['configuration'])
                measurement['results'] = json.loads(measurement['results'])
                results['utility_measurements'].append(measurement)
            
            # Get attack simulations
            cursor = self.connection.execute("""
                SELECT * FROM attack_simulations WHERE session_id = ?
                ORDER BY created_at DESC
            """, (session_id,))
            for row in cursor.fetchall():
                simulation = dict(row)
                simulation['configuration'] = json.loads(simulation['configuration'])
                simulation['results'] = json.loads(simulation['results'])
                results['attack_simulations'].append(simulation)
            
            # Get reports
            cursor = self.connection.execute("""
                SELECT * FROM generated_reports WHERE session_id = ?
                ORDER BY created_at DESC
            """, (session_id,))
            for row in cursor.fetchall():
                results['reports'].append(dict(row))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get session results: {str(e)}")
            return {}
    
    def get_dataset_metadata(self, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get dataset metadata"""
        
        try:
            if dataset_id:
                cursor = self.connection.execute("""
                    SELECT * FROM dataset_metadata WHERE dataset_id = ?
                """, (dataset_id,))
            else:
                cursor = self.connection.execute("""
                    SELECT * FROM dataset_metadata 
                    ORDER BY last_accessed DESC
                """)
            
            datasets = []
            for row in cursor.fetchall():
                dataset = dict(row)
                dataset['column_info'] = json.loads(dataset['column_info'])
                datasets.append(dataset)
            
            return datasets
            
        except Exception as e:
            self.logger.error(f"Failed to get dataset metadata: {str(e)}")
            return []
    
    def update_dataset_access(self, dataset_id: str):
        """Update last accessed time for dataset"""
        
        try:
            self.connection.execute("""
                UPDATE dataset_metadata 
                SET last_accessed = CURRENT_TIMESTAMP
                WHERE dataset_id = ?
            """, (dataset_id,))
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to update dataset access: {str(e)}")
    
    def get_application_logs(self, session_id: Optional[str] = None, 
                           level: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get application logs with optional filtering"""
        
        try:
            query = "SELECT * FROM application_logs WHERE 1=1"
            params = []
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            if level:
                query += " AND level = ?"
                params.append(level)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = self.connection.execute(query, params)
            
            logs = []
            for row in cursor.fetchall():
                logs.append(dict(row))
            
            return logs
            
        except Exception as e:
            self.logger.error(f"Failed to get application logs: {str(e)}")
            return []
    
    def get_user_preferences(self) -> Dict[str, str]:
        """Get all user preferences"""
        
        try:
            cursor = self.connection.execute("SELECT * FROM user_preferences")
            
            preferences = {}
            for row in cursor.fetchall():
                preferences[row['preference_key']] = row['preference_value']
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Failed to get user preferences: {str(e)}")
            return {}
    
    def set_user_preference(self, key: str, value: str):
        """Set a user preference"""
        
        try:
            self.connection.execute("""
                INSERT OR REPLACE INTO user_preferences 
                (preference_key, preference_value)
                VALUES (?, ?)
            """, (key, value))
            
            self.connection.commit()
            self.logger.debug(f"Set user preference: {key} = {value}")
            
        except Exception as e:
            self.logger.error(f"Failed to set user preference: {str(e)}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data beyond retention period"""
        
        try:
            # Delete old logs
            self.connection.execute("""
                DELETE FROM application_logs 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep))
            
            # Delete old sessions and related data
            old_sessions_cursor = self.connection.execute("""
                SELECT session_id FROM analysis_sessions 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days_to_keep))
            
            old_session_ids = [row[0] for row in old_sessions_cursor.fetchall()]
            
            for session_id in old_session_ids:
                # Delete related data
                self.connection.execute("""
                    DELETE FROM risk_assessments WHERE session_id = ?
                """, (session_id,))
                
                self.connection.execute("""
                    DELETE FROM privacy_enhancements WHERE session_id = ?
                """, (session_id,))
                
                self.connection.execute("""
                    DELETE FROM utility_measurements WHERE session_id = ?
                """, (session_id,))
                
                self.connection.execute("""
                    DELETE FROM attack_simulations WHERE session_id = ?
                """, (session_id,))
                
                self.connection.execute("""
                    DELETE FROM generated_reports WHERE session_id = ?
                """, (session_id,))
            
            # Delete old sessions
            self.connection.execute("""
                DELETE FROM analysis_sessions 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days_to_keep))
            
            self.connection.commit()
            
            deleted_count = len(old_session_ids)
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old analysis sessions")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {str(e)}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        try:
            stats = {}
            
            # Count records in each table
            tables = [
                'analysis_sessions', 'risk_assessments', 'privacy_enhancements',
                'utility_measurements', 'attack_simulations', 'application_logs',
                'dataset_metadata', 'generated_reports', 'user_preferences'
            ]
            
            for table in tables:
                cursor = self.connection.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Database file size
            if self.db_path.exists():
                stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            else:
                stats['database_size_mb'] = 0
            
            # Recent activity
            cursor = self.connection.execute("""
                SELECT COUNT(*) FROM analysis_sessions 
                WHERE created_at > datetime('now', '-7 days')
            """)
            stats['recent_sessions'] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {str(e)}")
            return {}
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create database backup"""
        
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = str(self.db_path.parent / f"safedata_backup_{timestamp}.db")
            
            # Use SQLite backup API
            backup_conn = sqlite3.connect(backup_path)
            self.connection.backup(backup_conn)
            backup_conn.close()
            
            self.logger.info(f"Database backed up to: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {str(e)}")
            raise
    
    def restore_database(self, backup_path: str):
        """Restore database from backup"""
        
        try:
            if not Path(backup_path).exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Close current connection
            if self.connection:
                self.connection.close()
            
            # Replace current database with backup
            import shutil
            shutil.copy2(backup_path, self.db_path)
            
            # Reconnect
            self.initialize_database()
            
            self.logger.info(f"Database restored from: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Database restore failed: {str(e)}")
            raise
    
    def close(self):
        """Close database connection"""
        
        if self.connection:
            try:
                self.connection.close()
                self.logger.debug("Database connection closed")
            except Exception as e:
                self.logger.error(f"Error closing database: {str(e)}")
    
    def __enter__(self):
        """Context manager entry"""
        if not self.connection:
            self.initialize_database()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
