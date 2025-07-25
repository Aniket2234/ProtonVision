"""
Attack Simulation Module for SafeData Pipeline
Simulates various types of linkage attacks to assess privacy risks
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class AttackSimulator:
    """Simulates various privacy attacks to assess dataset vulnerability"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def record_linkage_attack(self, dataset: pd.DataFrame, 
                             quasi_identifiers: List[str],
                             attack_knowledge_ratio: float = 0.1) -> Dict[str, Any]:
        """
        Simulate record linkage attack
        
        Args:
            dataset: Target dataset
            quasi_identifiers: List of QI columns
            attack_knowledge_ratio: Proportion of records attacker knows
            
        Returns:
            Attack results dictionary
        """
        self.logger.info("Simulating record linkage attack")
        
        try:
            if not quasi_identifiers or not all(col in dataset.columns for col in quasi_identifiers):
                return {
                    'success_rate': 0.0,
                    'attack_type': 'record_linkage',
                    'records_at_risk': 0,
                    'details': 'No valid quasi-identifiers found'
                }
            
            # Simulate attacker's auxiliary dataset
            n_known_records = int(len(dataset) * attack_knowledge_ratio)
            if n_known_records < 1:
                n_known_records = min(10, len(dataset))
            
            # Random sample as attacker's knowledge
            known_indices = np.random.choice(len(dataset), n_known_records, replace=False)
            auxiliary_data = dataset.iloc[known_indices][quasi_identifiers].copy()
            
            # Attempt to link records
            successful_links = 0
            total_attempts = 0
            link_confidences = []
            
            for idx, aux_record in auxiliary_data.iterrows():
                # Find matching records in dataset based on QI values
                matches = dataset[quasi_identifiers]
                
                # Calculate exact matches
                match_mask = (matches == aux_record).all(axis=1)
                matching_records = dataset[match_mask]
                
                total_attempts += 1
                
                if len(matching_records) == 1:
                    # Unique match - successful re-identification
                    successful_links += 1
                    link_confidences.append(1.0)
                elif len(matching_records) > 1:
                    # Multiple matches - partial success
                    confidence = 1.0 / len(matching_records)
                    link_confidences.append(confidence)
                    if confidence > 0.5:  # Consider high-confidence matches as successful
                        successful_links += 1
                else:
                    # No matches
                    link_confidences.append(0.0)
            
            success_rate = successful_links / total_attempts if total_attempts > 0 else 0.0
            
            # Calculate records at risk (those that can be uniquely identified)
            qi_combinations = dataset[quasi_identifiers].apply(tuple, axis=1)
            combination_counts = qi_combinations.value_counts()
            unique_records = (combination_counts == 1).sum()
            
            return {
                'success_rate': success_rate,
                'attack_type': 'record_linkage',
                'records_at_risk': int(unique_records),
                'total_attempts': total_attempts,
                'successful_links': successful_links,
                'average_confidence': np.mean(link_confidences) if link_confidences else 0.0,
                'unique_qi_combinations': len(combination_counts),
                'details': f'Successfully linked {successful_links}/{total_attempts} records'
            }
            
        except Exception as e:
            self.logger.error(f"Record linkage attack simulation failed: {str(e)}")
            return {
                'success_rate': 0.0,
                'attack_type': 'record_linkage',
                'records_at_risk': 0,
                'error': str(e)
            }
    
    def attribute_linkage_attack(self, dataset: pd.DataFrame, 
                               quasi_identifiers: List[str],
                               target_attribute: Optional[str] = None) -> Dict[str, Any]:
        """
        Simulate attribute linkage attack
        
        Args:
            dataset: Target dataset
            quasi_identifiers: List of QI columns
            target_attribute: Target sensitive attribute (default: last column)
            
        Returns:
            Attack results dictionary
        """
        self.logger.info("Simulating attribute linkage attack")
        
        try:
            if not quasi_identifiers or not all(col in dataset.columns for col in quasi_identifiers):
                return {
                    'success_rate': 0.0,
                    'attack_type': 'attribute_linkage',
                    'details': 'No valid quasi-identifiers found'
                }
            
            # Select target attribute
            if target_attribute is None or target_attribute not in dataset.columns:
                # Use last column as sensitive attribute
                sensitive_columns = [col for col in dataset.columns if col not in quasi_identifiers]
                if not sensitive_columns:
                    return {
                        'success_rate': 0.0,
                        'attack_type': 'attribute_linkage',
                        'details': 'No sensitive attributes found'
                    }
                target_attribute = sensitive_columns[-1]
            
            # Group by quasi-identifiers
            grouped = dataset.groupby(quasi_identifiers, dropna=False)
            
            successful_inferences = 0
            total_groups = 0
            inference_confidences = []
            
            for name, group in grouped:
                total_groups += 1
                
                # Check if sensitive attribute is homogeneous in this group
                sensitive_values = group[target_attribute].dropna()
                
                if len(sensitive_values) > 0:
                    value_counts = sensitive_values.value_counts()
                    most_common_count = value_counts.iloc[0]
                    total_count = len(sensitive_values)
                    
                    # Confidence is proportion of most common value
                    confidence = most_common_count / total_count
                    inference_confidences.append(confidence)
                    
                    # Consider high-confidence inferences as successful
                    if confidence > 0.8:
                        successful_inferences += 1
                else:
                    inference_confidences.append(0.0)
            
            success_rate = successful_inferences / total_groups if total_groups > 0 else 0.0
            
            # Calculate homogeneity risk
            homogeneity_risk = np.mean([c for c in inference_confidences if c > 0.5]) if inference_confidences else 0.0
            
            return {
                'success_rate': success_rate,
                'attack_type': 'attribute_linkage',
                'target_attribute': target_attribute,
                'vulnerable_groups': successful_inferences,
                'total_groups': total_groups,
                'average_confidence': np.mean(inference_confidences) if inference_confidences else 0.0,
                'homogeneity_risk': homogeneity_risk,
                'details': f'Successfully inferred attributes in {successful_inferences}/{total_groups} groups'
            }
            
        except Exception as e:
            self.logger.error(f"Attribute linkage attack simulation failed: {str(e)}")
            return {
                'success_rate': 0.0,
                'attack_type': 'attribute_linkage',
                'error': str(e)
            }
    
    def membership_inference_attack(self, dataset: pd.DataFrame,
                                  shadow_dataset_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Simulate membership inference attack
        
        Args:
            dataset: Target dataset
            shadow_dataset_ratio: Proportion of data to use for shadow model
            
        Returns:
            Attack results dictionary
        """
        self.logger.info("Simulating membership inference attack")
        
        try:
            if len(dataset) < 20:
                return {
                    'success_rate': 0.0,
                    'attack_type': 'membership_inference',
                    'details': 'Dataset too small for membership inference attack'
                }
            
            # Select numeric columns for the attack
            numeric_columns = dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) < 2:
                return {
                    'success_rate': 0.0,
                    'attack_type': 'membership_inference',
                    'details': 'Insufficient numeric features for attack'
                }
            
            # Split dataset for shadow model training
            shadow_size = int(len(dataset) * shadow_dataset_ratio)
            shadow_data = dataset.sample(n=shadow_size, random_state=42)
            target_data = dataset.drop(shadow_data.index)
            
            if len(target_data) < 5:
                return {
                    'success_rate': 0.0,
                    'attack_type': 'membership_inference',
                    'details': 'Insufficient data for attack simulation'
                }
            
            # Create training data for membership classifier
            # Positive examples: records in shadow dataset
            # Negative examples: synthetic "non-member" records
            
            # Generate synthetic non-member records
            synthetic_records = []
            for col in numeric_columns:
                col_mean = shadow_data[col].mean()
                col_std = shadow_data[col].std()
                if col_std > 0:
                    synthetic_col = np.random.normal(col_mean, col_std * 1.5, len(shadow_data))
                else:
                    synthetic_col = np.full(len(shadow_data), col_mean)
                synthetic_records.append(synthetic_col)
            
            synthetic_df = pd.DataFrame(
                np.column_stack(synthetic_records),
                columns=numeric_columns
            )
            
            # Prepare training data
            X_member = shadow_data[numeric_columns].fillna(shadow_data[numeric_columns].mean())
            X_non_member = synthetic_df.fillna(synthetic_df.mean())
            
            X_train = pd.concat([X_member, X_non_member])
            y_train = np.concatenate([np.ones(len(X_member)), np.zeros(len(X_non_member))])
            
            # Train membership classifier
            try:
                classifier = RandomForestClassifier(n_estimators=10, random_state=42)
                classifier.fit(X_train, y_train)
                
                # Test on target data (should be classified as members)
                X_test = target_data[numeric_columns].fillna(target_data[numeric_columns].mean())
                if len(X_test) > 0:
                    predictions = classifier.predict(X_test)
                    prediction_probs = classifier.predict_proba(X_test)[:, 1]  # Probability of membership
                    
                    # Success rate is proportion correctly identified as members
                    success_rate = np.mean(predictions)
                    average_confidence = np.mean(prediction_probs)
                    
                    # Calculate attack advantage over random guessing
                    random_baseline = 0.5
                    attack_advantage = success_rate - random_baseline
                    
                    return {
                        'success_rate': float(success_rate),
                        'attack_type': 'membership_inference',
                        'average_confidence': float(average_confidence),
                        'attack_advantage': float(attack_advantage),
                        'records_tested': len(X_test),
                        'correctly_identified': int(np.sum(predictions)),
                        'details': f'Identified {np.sum(predictions)}/{len(X_test)} members correctly'
                    }
                else:
                    return {
                        'success_rate': 0.0,
                        'attack_type': 'membership_inference',
                        'details': 'No test records available'
                    }
                    
            except Exception as e:
                self.logger.warning(f"Classifier training failed: {str(e)}")
                return {
                    'success_rate': 0.0,
                    'attack_type': 'membership_inference',
                    'details': f'Classifier training failed: {str(e)}'
                }
            
        except Exception as e:
            self.logger.error(f"Membership inference attack simulation failed: {str(e)}")
            return {
                'success_rate': 0.0,
                'attack_type': 'membership_inference',
                'error': str(e)
            }
    
    def homogeneity_attack(self, dataset: pd.DataFrame, 
                          quasi_identifiers: List[str],
                          sensitive_attribute: Optional[str] = None) -> Dict[str, Any]:
        """
        Simulate homogeneity attack (exploiting lack of l-diversity)
        
        Args:
            dataset: Target dataset
            quasi_identifiers: List of QI columns
            sensitive_attribute: Sensitive attribute (default: last non-QI column)
            
        Returns:
            Attack results dictionary
        """
        self.logger.info("Simulating homogeneity attack")
        
        try:
            if not quasi_identifiers or not all(col in dataset.columns for col in quasi_identifiers):
                return {
                    'success_rate': 0.0,
                    'attack_type': 'homogeneity',
                    'details': 'No valid quasi-identifiers found'
                }
            
            # Select sensitive attribute
            if sensitive_attribute is None or sensitive_attribute not in dataset.columns:
                sensitive_columns = [col for col in dataset.columns if col not in quasi_identifiers]
                if not sensitive_columns:
                    return {
                        'success_rate': 0.0,
                        'attack_type': 'homogeneity',
                        'details': 'No sensitive attributes found'
                    }
                sensitive_attribute = sensitive_columns[-1]
            
            # Group by quasi-identifiers
            grouped = dataset.groupby(quasi_identifiers, dropna=False)
            
            homogeneous_groups = 0
            total_groups = 0
            total_records_at_risk = 0
            diversity_scores = []
            
            for name, group in grouped:
                total_groups += 1
                group_size = len(group)
                
                # Calculate diversity of sensitive attribute in this group
                sensitive_values = group[sensitive_attribute].dropna()
                unique_values = len(sensitive_values.unique())
                
                if len(sensitive_values) > 0:
                    # Diversity score (0 = completely homogeneous, 1 = maximum diversity)
                    max_possible_diversity = min(len(sensitive_values), len(dataset[sensitive_attribute].unique()))
                    diversity_score = (unique_values - 1) / max(max_possible_diversity - 1, 1)
                    diversity_scores.append(diversity_score)
                    
                    # Group is vulnerable if it has low diversity (l-diversity < 2)
                    if unique_values <= 1:
                        homogeneous_groups += 1
                        total_records_at_risk += group_size
                else:
                    diversity_scores.append(0.0)
                    homogeneous_groups += 1
                    total_records_at_risk += group_size
            
            success_rate = homogeneous_groups / total_groups if total_groups > 0 else 0.0
            
            # Calculate overall risk metrics
            records_at_risk_ratio = total_records_at_risk / len(dataset)
            average_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
            
            # Assess l-diversity level
            min_l_diversity = 1
            for name, group in grouped:
                l_div = len(group[sensitive_attribute].dropna().unique())
                min_l_diversity = max(min_l_diversity, l_div)
            
            return {
                'success_rate': success_rate,
                'attack_type': 'homogeneity',
                'sensitive_attribute': sensitive_attribute,
                'vulnerable_groups': homogeneous_groups,
                'total_groups': total_groups,
                'records_at_risk': total_records_at_risk,
                'records_at_risk_ratio': records_at_risk_ratio,
                'average_diversity': average_diversity,
                'min_l_diversity': min_l_diversity,
                'details': f'{homogeneous_groups}/{total_groups} groups are homogeneous'
            }
            
        except Exception as e:
            self.logger.error(f"Homogeneity attack simulation failed: {str(e)}")
            return {
                'success_rate': 0.0,
                'attack_type': 'homogeneity',
                'error': str(e)
            }
    
    def background_knowledge_attack(self, dataset: pd.DataFrame,
                                  quasi_identifiers: List[str],
                                  background_knowledge_ratio: float = 0.2) -> Dict[str, Any]:
        """
        Simulate background knowledge attack
        
        Args:
            dataset: Target dataset
            quasi_identifiers: List of QI columns
            background_knowledge_ratio: Proportion of additional knowledge attacker has
            
        Returns:
            Attack results dictionary
        """
        self.logger.info("Simulating background knowledge attack")
        
        try:
            if not quasi_identifiers or not all(col in dataset.columns for col in quasi_identifiers):
                return {
                    'success_rate': 0.0,
                    'attack_type': 'background_knowledge',
                    'details': 'No valid quasi-identifiers found'
                }
            
            # Simulate attacker's background knowledge
            # This could include partial QI values, correlations, or external datasets
            
            # Create scenarios with different levels of background knowledge
            scenarios = []
            
            # Scenario 1: Partial QI knowledge
            if len(quasi_identifiers) > 1:
                partial_qi = quasi_identifiers[:max(1, len(quasi_identifiers)//2)]
                scenario_1 = self._test_partial_qi_attack(dataset, partial_qi, quasi_identifiers)
                scenarios.append(('partial_qi', scenario_1))
            
            # Scenario 2: Range knowledge (for numeric QIs)
            numeric_qis = [col for col in quasi_identifiers 
                          if pd.api.types.is_numeric_dtype(dataset[col])]
            if numeric_qis:
                scenario_2 = self._test_range_knowledge_attack(dataset, numeric_qis)
                scenarios.append(('range_knowledge', scenario_2))
            
            # Scenario 3: Correlation knowledge
            if len(quasi_identifiers) > 1:
                scenario_3 = self._test_correlation_attack(dataset, quasi_identifiers)
                scenarios.append(('correlation', scenario_3))
            
            if not scenarios:
                return {
                    'success_rate': 0.0,
                    'attack_type': 'background_knowledge',
                    'details': 'No applicable background knowledge scenarios'
                }
            
            # Aggregate results from all scenarios
            overall_success_rate = np.mean([result['success_rate'] for _, result in scenarios])
            
            return {
                'success_rate': overall_success_rate,
                'attack_type': 'background_knowledge',
                'scenarios': dict(scenarios),
                'total_scenarios': len(scenarios),
                'details': f'Tested {len(scenarios)} background knowledge scenarios'
            }
            
        except Exception as e:
            self.logger.error(f"Background knowledge attack simulation failed: {str(e)}")
            return {
                'success_rate': 0.0,
                'attack_type': 'background_knowledge',
                'error': str(e)
            }
    
    def _test_partial_qi_attack(self, dataset: pd.DataFrame, 
                               partial_qi: List[str], 
                               full_qi: List[str]) -> Dict[str, Any]:
        """Test attack using partial quasi-identifier knowledge"""
        
        try:
            # Group by partial QIs
            partial_groups = dataset.groupby(partial_qi, dropna=False)
            
            successful_narrows = 0
            total_attempts = 0
            
            for name, group in partial_groups:
                total_attempts += 1
                
                # Check if partial knowledge significantly narrows down possibilities
                group_size = len(group)
                dataset_size = len(dataset)
                
                # Success if group size is small relative to dataset
                if group_size <= max(5, dataset_size * 0.05):
                    successful_narrows += 1
            
            success_rate = successful_narrows / total_attempts if total_attempts > 0 else 0.0
            
            return {
                'success_rate': success_rate,
                'method': 'partial_qi',
                'successful_narrows': successful_narrows,
                'total_attempts': total_attempts
            }
            
        except Exception:
            return {'success_rate': 0.0, 'method': 'partial_qi'}
    
    def _test_range_knowledge_attack(self, dataset: pd.DataFrame, 
                                   numeric_qis: List[str]) -> Dict[str, Any]:
        """Test attack using range knowledge of numeric QIs"""
        
        try:
            # Simulate attacker knowing value ranges instead of exact values
            successful_narrows = 0
            total_tests = min(50, len(dataset))  # Limit tests for performance
            
            test_indices = np.random.choice(len(dataset), total_tests, replace=False)
            
            for idx in test_indices:
                record = dataset.iloc[idx]
                candidate_count = len(dataset)
                
                # Apply range constraints for each numeric QI
                for col in numeric_qis:
                    value = record[col]
                    if not pd.isna(value):
                        # Assume attacker knows value within ±10% range
                        range_width = abs(value * 0.1) if value != 0 else 1
                        lower_bound = value - range_width
                        upper_bound = value + range_width
                        
                        # Count candidates within range
                        in_range = ((dataset[col] >= lower_bound) & 
                                   (dataset[col] <= upper_bound))
                        candidate_count = min(candidate_count, in_range.sum())
                
                # Success if significantly narrowed down
                if candidate_count <= max(3, len(dataset) * 0.01):
                    successful_narrows += 1
            
            success_rate = successful_narrows / total_tests if total_tests > 0 else 0.0
            
            return {
                'success_rate': success_rate,
                'method': 'range_knowledge',
                'successful_narrows': successful_narrows,
                'total_tests': total_tests
            }
            
        except Exception:
            return {'success_rate': 0.0, 'method': 'range_knowledge'}
    
    def _test_correlation_attack(self, dataset: pd.DataFrame, 
                               quasi_identifiers: List[str]) -> Dict[str, Any]:
        """Test attack using correlation knowledge between QIs"""
        
        try:
            # Check correlations between QIs
            numeric_qis = [col for col in quasi_identifiers 
                          if pd.api.types.is_numeric_dtype(dataset[col])]
            
            if len(numeric_qis) < 2:
                return {'success_rate': 0.0, 'method': 'correlation'}
            
            # Calculate correlation matrix
            corr_matrix = dataset[numeric_qis].corr().abs()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(numeric_qis)):
                for j in range(i+1, len(numeric_qis)):
                    corr_value = corr_matrix.iloc[i, j]
                    if corr_value > 0.7:  # High correlation threshold
                        high_corr_pairs.append((numeric_qis[i], numeric_qis[j], corr_value))
            
            if not high_corr_pairs:
                return {'success_rate': 0.0, 'method': 'correlation'}
            
            # Test if knowledge of one QI helps predict another
            successful_predictions = 0
            total_tests = 0
            
            for col1, col2, corr in high_corr_pairs:
                # Simple prediction test: use correlation to estimate values
                test_size = min(20, len(dataset))
                test_indices = np.random.choice(len(dataset), test_size, replace=False)
                
                for idx in test_indices:
                    actual_val2 = dataset.iloc[idx][col2]
                    known_val1 = dataset.iloc[idx][col1]
                    
                    if not (pd.isna(actual_val2) or pd.isna(known_val1)):
                        # Predict val2 based on correlation with val1
                        # Simple linear prediction
                        mean1, mean2 = dataset[col1].mean(), dataset[col2].mean()
                        std1, std2 = dataset[col1].std(), dataset[col2].std()
                        
                        if std1 > 0:
                            predicted_val2 = mean2 + corr * (std2/std1) * (known_val1 - mean1)
                            
                            # Check if prediction is close (within 1 standard deviation)
                            if abs(predicted_val2 - actual_val2) <= std2:
                                successful_predictions += 1
                        
                        total_tests += 1
            
            success_rate = successful_predictions / total_tests if total_tests > 0 else 0.0
            
            return {
                'success_rate': success_rate,
                'method': 'correlation',
                'high_corr_pairs': len(high_corr_pairs),
                'successful_predictions': successful_predictions,
                'total_tests': total_tests
            }
            
        except Exception:
            return {'success_rate': 0.0, 'method': 'correlation'}
    
    def temporal_attack(self, dataset: pd.DataFrame, 
                       quasi_identifiers: List[str],
                       temporal_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Simulate temporal attack using time-based patterns
        
        Args:
            dataset: Target dataset
            quasi_identifiers: List of QI columns
            temporal_column: Column containing temporal information
            
        Returns:
            Attack results dictionary
        """
        self.logger.info("Simulating temporal attack")
        
        try:
            # Find temporal column if not specified
            if temporal_column is None:
                datetime_columns = dataset.select_dtypes(include=['datetime64']).columns
                if len(datetime_columns) > 0:
                    temporal_column = datetime_columns[0]
                else:
                    # Look for columns that might contain temporal patterns
                    for col in dataset.columns:
                        if any(word in col.lower() for word in ['date', 'time', 'year', 'month']):
                            temporal_column = col
                            break
            
            if temporal_column is None or temporal_column not in dataset.columns:
                return {
                    'success_rate': 0.0,
                    'attack_type': 'temporal',
                    'details': 'No temporal column found'
                }
            
            # Analyze temporal patterns
            if pd.api.types.is_datetime64_any_dtype(dataset[temporal_column]):
                # Extract temporal features
                dataset_copy = dataset.copy()
                dataset_copy['year'] = dataset_copy[temporal_column].dt.year
                dataset_copy['month'] = dataset_copy[temporal_column].dt.month
                dataset_copy['day'] = dataset_copy[temporal_column].dt.day
                temporal_features = ['year', 'month', 'day']
            elif pd.api.types.is_numeric_dtype(dataset[temporal_column]):
                # Assume numeric temporal data (e.g., year)
                temporal_features = [temporal_column]
                dataset_copy = dataset.copy()
            else:
                return {
                    'success_rate': 0.0,
                    'attack_type': 'temporal',
                    'details': 'Temporal column is not datetime or numeric'
                }
            
            # Test temporal linkage attack
            extended_qi = quasi_identifiers + temporal_features
            
            # Group by extended QIs (including temporal features)
            temporal_groups = dataset_copy.groupby(extended_qi, dropna=False).size()
            
            # Count unique identifications
            unique_temporal_records = (temporal_groups == 1).sum()
            temporal_success_rate = unique_temporal_records / len(dataset)
            
            # Compare with non-temporal attack
            base_groups = dataset_copy.groupby(quasi_identifiers, dropna=False).size()
            unique_base_records = (base_groups == 1).sum()
            base_success_rate = unique_base_records / len(dataset)
            
            # Calculate temporal attack advantage
            temporal_advantage = temporal_success_rate - base_success_rate
            
            return {
                'success_rate': float(temporal_success_rate),
                'attack_type': 'temporal',
                'temporal_column': temporal_column,
                'base_success_rate': float(base_success_rate),
                'temporal_advantage': float(temporal_advantage),
                'unique_temporal_records': int(unique_temporal_records),
                'unique_base_records': int(unique_base_records),
                'details': f'Temporal features increased attack success by {temporal_advantage:.1%}'
            }
            
        except Exception as e:
            self.logger.error(f"Temporal attack simulation failed: {str(e)}")
            return {
                'success_rate': 0.0,
                'attack_type': 'temporal',
                'error': str(e)
            }
    
    def composition_attack(self, datasets: List[pd.DataFrame],
                          common_identifiers: List[str]) -> Dict[str, Any]:
        """
        Simulate composition attack across multiple datasets
        
        Args:
            datasets: List of datasets to link
            common_identifiers: Common identifier columns across datasets
            
        Returns:
            Attack results dictionary
        """
        self.logger.info("Simulating composition attack")
        
        try:
            if len(datasets) < 2:
                return {
                    'success_rate': 0.0,
                    'attack_type': 'composition',
                    'details': 'Need at least 2 datasets for composition attack'
                }
            
            # Verify common identifiers exist in all datasets
            valid_identifiers = []
            for identifier in common_identifiers:
                if all(identifier in df.columns for df in datasets):
                    valid_identifiers.append(identifier)
            
            if not valid_identifiers:
                return {
                    'success_rate': 0.0,
                    'attack_type': 'composition',
                    'details': 'No common identifiers found across datasets'
                }
            
            # Start with first dataset
            linked_data = datasets[0][valid_identifiers].copy()
            linked_data['dataset_0'] = True
            
            successful_links = 0
            total_records = len(linked_data)
            
            # Progressively link additional datasets
            for i, dataset in enumerate(datasets[1:], 1):
                # Merge with current linked data
                dataset_subset = dataset[valid_identifiers].copy()
                dataset_subset[f'dataset_{i}'] = True
                
                # Perform inner join to find common records
                merged = pd.merge(linked_data, dataset_subset, on=valid_identifiers, how='inner')
                
                # Count successful links
                links_found = len(merged)
                successful_links += links_found
                
                # Update linked data for next iteration
                linked_data = merged
            
            # Calculate success rate
            if len(datasets) > 1:
                max_possible_links = total_records * (len(datasets) - 1)
                success_rate = successful_links / max_possible_links if max_possible_links > 0 else 0.0
            else:
                success_rate = 0.0
            
            # Analyze final linked dataset
            final_unique_records = len(linked_data)
            linkage_ratio = final_unique_records / total_records if total_records > 0 else 0.0
            
            return {
                'success_rate': float(success_rate),
                'attack_type': 'composition',
                'datasets_count': len(datasets),
                'common_identifiers': valid_identifiers,
                'successful_links': successful_links,
                'final_linked_records': final_unique_records,
                'linkage_ratio': float(linkage_ratio),
                'details': f'Successfully linked {final_unique_records} records across {len(datasets)} datasets'
            }
            
        except Exception as e:
            self.logger.error(f"Composition attack simulation failed: {str(e)}")
            return {
                'success_rate': 0.0,
                'attack_type': 'composition',
                'error': str(e)
            }
    
    def assess_overall_vulnerability(self, dataset: pd.DataFrame,
                                   quasi_identifiers: List[str],
                                   attack_types: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive vulnerability assessment
        
        Args:
            dataset: Target dataset
            quasi_identifiers: List of QI columns
            attack_types: List of attack types to test
            
        Returns:
            Comprehensive vulnerability assessment
        """
        if attack_types is None:
            attack_types = ['record_linkage', 'attribute_linkage', 'membership_inference', 'homogeneity']
        
        self.logger.info(f"Performing comprehensive vulnerability assessment with {len(attack_types)} attack types")
        
        results = {}
        overall_scores = []
        
        try:
            # Run each specified attack
            for attack_type in attack_types:
                if attack_type == 'record_linkage':
                    result = self.record_linkage_attack(dataset, quasi_identifiers)
                elif attack_type == 'attribute_linkage':
                    result = self.attribute_linkage_attack(dataset, quasi_identifiers)
                elif attack_type == 'membership_inference':
                    result = self.membership_inference_attack(dataset)
                elif attack_type == 'homogeneity':
                    result = self.homogeneity_attack(dataset, quasi_identifiers)
                elif attack_type == 'background_knowledge':
                    result = self.background_knowledge_attack(dataset, quasi_identifiers)
                elif attack_type == 'temporal':
                    result = self.temporal_attack(dataset, quasi_identifiers)
                else:
                    self.logger.warning(f"Unknown attack type: {attack_type}")
                    continue
                
                results[attack_type] = result
                success_rate = result.get('success_rate', 0.0)
                overall_scores.append(success_rate)
            
            # Calculate overall vulnerability score
            overall_vulnerability = np.mean(overall_scores) if overall_scores else 0.0
            
            # Determine risk level
            if overall_vulnerability < 0.1:
                risk_level = "Low"
            elif overall_vulnerability < 0.3:
                risk_level = "Medium"
            elif overall_vulnerability < 0.7:
                risk_level = "High"
            else:
                risk_level = "Critical"
            
            # Generate recommendations
            recommendations = self._generate_attack_recommendations(results, overall_vulnerability)
            
            return {
                'overall_vulnerability': float(overall_vulnerability),
                'risk_level': risk_level,
                'attack_results': results,
                'tested_attacks': len(attack_types),
                'successful_attacks': sum(1 for score in overall_scores if score > 0.1),
                'recommendations': recommendations,
                'assessment_summary': {
                    'highest_risk_attack': max(results.items(), key=lambda x: x[1].get('success_rate', 0))[0] if results else None,
                    'average_success_rate': float(overall_vulnerability),
                    'total_attacks_tested': len(attack_types)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Overall vulnerability assessment failed: {str(e)}")
            return {
                'overall_vulnerability': 0.0,
                'risk_level': "Unknown",
                'attack_results': results,
                'error': str(e)
            }
    
    def _generate_attack_recommendations(self, attack_results: Dict[str, Any], 
                                       overall_vulnerability: float) -> List[str]:
        """Generate recommendations based on attack results"""
        
        recommendations = []
        
        # Overall recommendations
        if overall_vulnerability > 0.5:
            recommendations.append("Critical: Dataset shows high vulnerability to multiple attacks")
            recommendations.append("Apply strong privacy-preserving techniques immediately")
        elif overall_vulnerability > 0.3:
            recommendations.append("Moderate risk detected - enhance privacy protection")
        
        # Specific attack recommendations
        for attack_type, result in attack_results.items():
            success_rate = result.get('success_rate', 0.0)
            
            if success_rate > 0.3:
                if attack_type == 'record_linkage':
                    recommendations.append("High record linkage risk - increase k-anonymity")
                elif attack_type == 'attribute_linkage':
                    recommendations.append("High attribute inference risk - improve l-diversity")
                elif attack_type == 'membership_inference':
                    recommendations.append("Membership inference vulnerability - apply differential privacy")
                elif attack_type == 'homogeneity':
                    recommendations.append("Homogeneity attack risk - ensure diverse sensitive values")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Low attack vulnerability detected - maintain current privacy level")
        
        recommendations.append("Regular privacy assessment recommended")
        
        return recommendations[:10]  # Limit to top 10 recommendations
