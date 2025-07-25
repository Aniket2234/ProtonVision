"""
Risk Assessment Module for SafeData Pipeline
Evaluates re-identification risks through simulated linkage attacks
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from collections import Counter
import itertools

from utils.privacy_metrics import PrivacyMetrics
from utils.attack_simulation import AttackSimulator

class RiskAssessment:
    """Main class for privacy risk assessment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.privacy_metrics = PrivacyMetrics()
        self.attack_simulator = AttackSimulator()
        
    def assess_dataset(self, dataset: pd.DataFrame, config: Dict) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment on dataset
        
        Args:
            dataset: Input dataset to assess
            config: Configuration parameters
            
        Returns:
            Dictionary containing risk assessment results
        """
        self.logger.info("Starting risk assessment")
        
        try:
            # Step 1: Identify quasi-identifiers
            quasi_identifiers = self._identify_quasi_identifiers(dataset, config)
            
            # Step 2: Calculate individual risks
            individual_risks = self._calculate_individual_risks(dataset, quasi_identifiers)
            
            # Step 3: Calculate global risk metrics
            global_risk = self._calculate_global_risk(individual_risks)
            
            # Step 4: Perform linkage attack simulations
            attack_results = self._simulate_attacks(dataset, quasi_identifiers, config)
            
            # Step 5: Calculate prosecutor and journalist risks
            prosecutor_risk = self._calculate_prosecutor_risk(individual_risks)
            journalist_risk = self._calculate_journalist_risk(individual_risks)
            
            # Step 6: Assess k-anonymity and l-diversity
            anonymity_metrics = self._assess_anonymity_metrics(dataset, quasi_identifiers)
            
            # Compile results
            results = {
                'quasi_identifiers': quasi_identifiers,
                'individual_risks': individual_risks,
                'global_risk': global_risk,
                'prosecutor_risk': prosecutor_risk,
                'journalist_risk': journalist_risk,
                'attack_results': attack_results,
                'anonymity_metrics': anonymity_metrics,
                'risk_distribution': self._analyze_risk_distribution(individual_risks),
                'high_risk_records': self._identify_high_risk_records(individual_risks, config),
                'compliance_status': self._assess_compliance(global_risk, config)
            }
            
            self.logger.info("Risk assessment completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {str(e)}")
            raise
    
    def _identify_quasi_identifiers(self, dataset: pd.DataFrame, config: Dict) -> List[str]:
        """Identify quasi-identifier columns in the dataset"""
        
        if config.get('auto_detect_qi', True):
            # Automatic detection based on data characteristics
            quasi_identifiers = []
            
            for column in dataset.columns:
                # Check if column might be a quasi-identifier
                if self._is_potential_qi(dataset[column]):
                    quasi_identifiers.append(column)
                    
        else:
            # Use manually specified quasi-identifiers
            quasi_identifiers = config.get('quasi_identifiers', [])
            
        self.logger.info(f"Identified quasi-identifiers: {quasi_identifiers}")
        return quasi_identifiers
    
    def _is_potential_qi(self, series: pd.Series) -> bool:
        """Determine if a column is a potential quasi-identifier"""
        
        # Check uniqueness ratio
        unique_ratio = series.nunique() / len(series)
        
        # Check data type and characteristics
        if series.dtype in ['object', 'category']:
            # Categorical data with moderate uniqueness
            return 0.1 < unique_ratio < 0.8
        elif pd.api.types.is_numeric_dtype(series):
            # Numeric data that might be age, income, etc.
            return unique_ratio > 0.05
        elif pd.api.types.is_datetime64_any_dtype(series):
            # Date columns are often quasi-identifiers
            return True
            
        return False
    
    def _calculate_individual_risks(self, dataset: pd.DataFrame, quasi_identifiers: List[str]) -> np.ndarray:
        """Calculate re-identification risk for each individual record"""
        
        if not quasi_identifiers:
            # If no QIs, assume low uniform risk
            return np.full(len(dataset), 0.01)
        
        # Create QI combinations for each record
        qi_combinations = dataset[quasi_identifiers].apply(
            lambda row: tuple(row.values), axis=1
        )
        
        # Count frequency of each combination
        combination_counts = qi_combinations.value_counts()
        
        # Calculate risk as 1/frequency for each record
        individual_risks = qi_combinations.map(lambda x: 1.0 / combination_counts[x])
        
        return individual_risks.values
    
    def _calculate_global_risk(self, individual_risks: np.ndarray) -> float:
        """Calculate global re-identification risk"""
        
        # Global risk as average of individual risks
        global_risk = np.mean(individual_risks)
        
        return min(global_risk, 1.0)  # Cap at 1.0
    
    def _simulate_attacks(self, dataset: pd.DataFrame, quasi_identifiers: List[str], config: Dict) -> Dict[str, Any]:
        """Simulate various types of linkage attacks"""
        
        attack_types = config.get('attack_types', [])
        attack_results = {}
        
        for attack_type in attack_types:
            try:
                if attack_type == "Record Linkage":
                    result = self.attack_simulator.record_linkage_attack(dataset, quasi_identifiers)
                elif attack_type == "Attribute Linkage":
                    result = self.attack_simulator.attribute_linkage_attack(dataset, quasi_identifiers)
                elif attack_type == "Membership Inference":
                    result = self.attack_simulator.membership_inference_attack(dataset)
                elif attack_type == "Homogeneity Attack":
                    result = self.attack_simulator.homogeneity_attack(dataset, quasi_identifiers)
                else:
                    result = {"success_rate": 0.0, "details": "Unknown attack type"}
                    
                attack_results[attack_type] = result
                
            except Exception as e:
                self.logger.warning(f"Attack simulation failed for {attack_type}: {str(e)}")
                attack_results[attack_type] = {"success_rate": 0.0, "error": str(e)}
        
        return attack_results
    
    def _calculate_prosecutor_risk(self, individual_risks: np.ndarray) -> float:
        """Calculate prosecutor risk (maximum individual risk)"""
        return np.max(individual_risks) if len(individual_risks) > 0 else 0.0
    
    def _calculate_journalist_risk(self, individual_risks: np.ndarray) -> float:
        """Calculate journalist risk (expected re-identification rate)"""
        return np.mean(individual_risks) if len(individual_risks) > 0 else 0.0
    
    def _assess_anonymity_metrics(self, dataset: pd.DataFrame, quasi_identifiers: List[str]) -> Dict[str, Any]:
        """Assess k-anonymity, l-diversity, and t-closeness"""
        
        if not quasi_identifiers:
            return {
                'k_anonymity': len(dataset),
                'l_diversity': 1,
                't_closeness': 0.0,
                'groups': 1
            }
        
        # Group by quasi-identifiers
        grouped = dataset.groupby(quasi_identifiers)
        group_sizes = grouped.size()
        
        # K-anonymity: minimum group size
        k_anonymity = int(group_sizes.min())
        
        # L-diversity: minimum number of distinct sensitive values per group
        # For simplicity, assume last column is sensitive attribute
        sensitive_attr = dataset.columns[-1]
        l_diversity = grouped[sensitive_attr].nunique().min() if sensitive_attr in dataset.columns else 1
        
        # T-closeness: simplified calculation
        t_closeness = self._calculate_t_closeness(dataset, quasi_identifiers, sensitive_attr)
        
        return {
            'k_anonymity': k_anonymity,
            'l_diversity': int(l_diversity),
            't_closeness': t_closeness,
            'groups': len(group_sizes),
            'group_sizes': group_sizes.tolist()
        }
    
    def _calculate_t_closeness(self, dataset: pd.DataFrame, quasi_identifiers: List[str], sensitive_attr: str) -> float:
        """Calculate t-closeness metric"""
        
        try:
            if sensitive_attr not in dataset.columns:
                return 0.0
            
            # Global distribution of sensitive attribute
            global_dist = dataset[sensitive_attr].value_counts(normalize=True)
            
            # Group by quasi-identifiers
            grouped = dataset.groupby(quasi_identifiers)
            
            max_distance = 0.0
            for name, group in grouped:
                if len(group) > 0:
                    # Local distribution
                    local_dist = group[sensitive_attr].value_counts(normalize=True)
                    
                    # Calculate Earth Mover's Distance (simplified as L1 distance)
                    all_values = set(global_dist.index) | set(local_dist.index)
                    distance = sum(abs(global_dist.get(val, 0) - local_dist.get(val, 0)) for val in all_values) / 2
                    
                    max_distance = max(max_distance, distance)
            
            return max_distance
            
        except Exception:
            return 0.0
    
    def _analyze_risk_distribution(self, individual_risks: np.ndarray) -> Dict[str, Any]:
        """Analyze distribution of individual risks"""
        
        if len(individual_risks) == 0:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'percentiles': {25: 0.0, 75: 0.0, 95: 0.0, 99: 0.0}
            }
        
        return {
            'mean': float(np.mean(individual_risks)),
            'median': float(np.median(individual_risks)),
            'std': float(np.std(individual_risks)),
            'min': float(np.min(individual_risks)),
            'max': float(np.max(individual_risks)),
            'percentiles': {
                25: float(np.percentile(individual_risks, 25)),
                75: float(np.percentile(individual_risks, 75)),
                95: float(np.percentile(individual_risks, 95)),
                99: float(np.percentile(individual_risks, 99))
            }
        }
    
    def _identify_high_risk_records(self, individual_risks: np.ndarray, config: Dict) -> List[int]:
        """Identify records with high re-identification risk"""
        
        threshold = config.get('individual_threshold', 0.1)
        high_risk_indices = np.where(individual_risks > threshold)[0]
        
        return high_risk_indices.tolist()
    
    def _assess_compliance(self, global_risk: float, config: Dict) -> Dict[str, Any]:
        """Assess compliance with privacy regulations"""
        
        global_threshold = config.get('global_threshold', 0.05)
        individual_threshold = config.get('individual_threshold', 0.1)
        
        # DPDP Act compliance assessment
        dpdp_compliant = global_risk <= global_threshold
        
        # General privacy compliance
        privacy_level = (
            "High" if global_risk < 0.01 else
            "Medium" if global_risk < 0.05 else
            "Low"
        )
        
        recommendations = []
        if global_risk > global_threshold:
            recommendations.append("Apply additional privacy-preserving techniques")
        if global_risk > 0.1:
            recommendations.append("Consider synthetic data generation")
        if global_risk > 0.5:
            recommendations.append("Dataset requires significant privacy enhancement")
        
        return {
            'dpdp_compliant': dpdp_compliant,
            'privacy_level': privacy_level,
            'global_risk': global_risk,
            'threshold_met': global_risk <= global_threshold,
            'recommendations': recommendations
        }
