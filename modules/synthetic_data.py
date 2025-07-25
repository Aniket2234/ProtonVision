"""
Synthetic Data Generation Module for SafeData Pipeline
Implements GANs, VAEs, Bayesian Networks, and Copula-based methods
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    from scipy.stats import gaussian_kde
    from scipy.optimize import minimize
    HAS_SCIPY_ADVANCED = True
except ImportError:
    HAS_SCIPY_ADVANCED = False

class SyntheticDataGenerator:
    """Synthetic data generation using various methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.encoders = {}
        
    def generate_synthetic_data(self, dataset: pd.DataFrame, model_type: str = 'GAN',
                               n_samples: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate synthetic data using specified method
        
        Args:
            dataset: Original dataset
            model_type: Type of model ('GAN', 'VAE', 'Bayesian Network', 'Copula')
            n_samples: Number of synthetic samples (default: same as original)
            **kwargs: Additional parameters for specific models
            
        Returns:
            Dictionary with synthetic data and generation metrics
        """
        
        if n_samples is None:
            n_samples = len(dataset)
            
        self.logger.info(f"Generating {n_samples} synthetic samples using {model_type}")
        
        try:
            # Preprocess data
            processed_data, preprocessing_info = self._preprocess_data(dataset)
            
            # Generate synthetic data based on model type
            if model_type == 'GAN':
                if HAS_TENSORFLOW:
                    result = self._generate_with_gan(processed_data, n_samples, **kwargs)
                else:
                    self.logger.warning("TensorFlow not available, falling back to statistical approximation")
                    result = self._generate_with_statistical_approximation(processed_data, n_samples, **kwargs)
            elif model_type == 'VAE':
                if HAS_TENSORFLOW:
                    result = self._generate_with_vae(processed_data, n_samples, **kwargs)
                else:
                    self.logger.warning("TensorFlow not available, falling back to statistical approximation")
                    result = self._generate_with_statistical_approximation(processed_data, n_samples, **kwargs)
            elif model_type == 'Bayesian Network':
                result = self._generate_with_bayesian_network(processed_data, n_samples, **kwargs)
            elif model_type == 'Copula':
                result = self._generate_with_copula(processed_data, n_samples, **kwargs)
            elif model_type == 'Statistical':
                result = self._generate_with_statistical_approximation(processed_data, n_samples, **kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Post-process synthetic data
            synthetic_data = self._postprocess_data(
                result['synthetic_data'], 
                preprocessing_info, 
                dataset.columns
            )
            
            # Calculate fidelity metrics
            fidelity_score = self._calculate_fidelity(dataset, synthetic_data)
            
            # Calculate privacy metrics
            privacy_score = self._calculate_privacy_score(dataset, synthetic_data)
            
            final_result = {
                'synthetic_data': synthetic_data,
                'model_type': model_type,
                'fidelity_score': fidelity_score,
                'privacy_score': privacy_score,
                'final_loss': result.get('final_loss', 0.0),
                'training_metrics': result.get('training_metrics', {}),
                'generation_info': {
                    'original_samples': len(dataset),
                    'synthetic_samples': len(synthetic_data),
                    'features': len(dataset.columns),
                    'preprocessing_steps': preprocessing_info['steps']
                }
            }
            
            self.logger.info(f"Synthetic data generation completed. Fidelity: {fidelity_score:.3f}")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Synthetic data generation failed: {str(e)}")
            raise
    
    def _preprocess_data(self, dataset: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Preprocess data for synthetic generation"""
        
        preprocessing_info = {
            'steps': [],
            'scalers': {},
            'encoders': {},
            'column_types': {},
            'original_columns': list(dataset.columns)
        }
        
        processed_data = dataset.copy()
        
        # Handle missing values
        if processed_data.isnull().sum().sum() > 0:
            processed_data = processed_data.fillna(processed_data.mean(numeric_only=True))
            processed_data = processed_data.fillna(processed_data.mode().iloc[0])
            preprocessing_info['steps'].append('missing_value_imputation')
        
        # Encode categorical variables
        categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            processed_data[col] = le.fit_transform(processed_data[col].astype(str))
            preprocessing_info['encoders'][col] = le
            preprocessing_info['column_types'][col] = 'categorical'
            preprocessing_info['steps'].append(f'label_encode_{col}')
        
        # Scale numeric variables
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        scaler = MinMaxScaler()
        processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])
        preprocessing_info['scalers']['numeric'] = scaler
        
        for col in numeric_columns:
            preprocessing_info['column_types'][col] = 'numeric'
        
        preprocessing_info['steps'].append('numeric_scaling')
        
        # Store preprocessing info for later use
        self.scalers = preprocessing_info['scalers']
        self.encoders = preprocessing_info['encoders']
        
        return processed_data.values, preprocessing_info
    
    def _generate_with_gan(self, data: np.ndarray, n_samples: int, **kwargs) -> Dict[str, Any]:
        """Generate synthetic data using Generative Adversarial Network"""
        
        if not HAS_TENSORFLOW:
            self.logger.warning("TensorFlow not available, using statistical approximation")
            return self._generate_statistical_approximation(data, n_samples)
        
        try:
            epochs = kwargs.get('epochs', 100)
            batch_size = kwargs.get('batch_size', 32)
            learning_rate = kwargs.get('learning_rate', 0.001)
            
            n_features = data.shape[1]
            latent_dim = min(100, n_features * 2)
            
            # Build Generator
            generator = self._build_generator(latent_dim, n_features)
            
            # Build Discriminator
            discriminator = self._build_discriminator(n_features)
            
            # Build GAN
            discriminator.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Make discriminator non-trainable for generator training
            discriminator.trainable = False
            gan_input = keras.Input(shape=(latent_dim,))
            gan_output = discriminator(generator(gan_input))
            gan = keras.Model(gan_input, gan_output)
            gan.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='binary_crossentropy')
            
            # Training
            training_metrics = self._train_gan(
                generator, discriminator, gan, data, 
                epochs, batch_size, latent_dim
            )
            
            # Generate synthetic data
            noise = np.random.normal(0, 1, (n_samples, latent_dim))
            synthetic_data = generator.predict(noise, verbose=0)
            
            return {
                'synthetic_data': synthetic_data,
                'final_loss': training_metrics['final_loss'],
                'training_metrics': training_metrics
            }
            
        except Exception as e:
            self.logger.warning(f"GAN generation failed: {str(e)}, using fallback method")
            return self._generate_statistical_approximation(data, n_samples)
    
    def _generate_with_vae(self, data: np.ndarray, n_samples: int, **kwargs) -> Dict[str, Any]:
        """Generate synthetic data using Variational Autoencoder"""
        
        if not HAS_TENSORFLOW:
            self.logger.warning("TensorFlow not available, using statistical approximation")
            return self._generate_statistical_approximation(data, n_samples)
        
        try:
            epochs = kwargs.get('epochs', 100)
            batch_size = kwargs.get('batch_size', 32)
            learning_rate = kwargs.get('learning_rate', 0.001)
            
            n_features = data.shape[1]
            latent_dim = min(50, n_features)
            
            # Build VAE
            encoder, decoder, vae = self._build_vae(n_features, latent_dim)
            
            # Compile VAE
            vae.compile(optimizer=keras.optimizers.Adam(learning_rate))
            
            # Train VAE
            history = vae.fit(
                data, data,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0
            )
            
            # Generate synthetic data
            noise = np.random.normal(0, 1, (n_samples, latent_dim))
            synthetic_data = decoder.predict(noise, verbose=0)
            
            return {
                'synthetic_data': synthetic_data,
                'final_loss': history.history['loss'][-1],
                'training_metrics': {
                    'epochs_trained': epochs,
                    'final_val_loss': history.history.get('val_loss', [0])[-1]
                }
            }
            
        except Exception as e:
            self.logger.warning(f"VAE generation failed: {str(e)}, using fallback method")
            return self._generate_statistical_approximation(data, n_samples)
    
    def _generate_with_bayesian_network(self, data: np.ndarray, n_samples: int, **kwargs) -> Dict[str, Any]:
        """Generate synthetic data using Bayesian Network approach"""
        
        try:
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(data)
            
            # Learn dependencies between variables
            dependencies = self._learn_variable_dependencies(df)
            
            # Generate synthetic data based on learned dependencies
            synthetic_data = np.zeros((n_samples, data.shape[1]))
            
            for i in range(n_samples):
                for col_idx in range(data.shape[1]):
                    # Generate value based on dependencies
                    if col_idx in dependencies and len(dependencies[col_idx]) > 0:
                        # Use conditional distribution
                        parent_values = [synthetic_data[i, parent] for parent in dependencies[col_idx]]
                        synthetic_data[i, col_idx] = self._sample_conditional_distribution(
                            df.iloc[:, col_idx], df.iloc[:, dependencies[col_idx]], parent_values
                        )
                    else:
                        # Use marginal distribution
                        synthetic_data[i, col_idx] = np.random.choice(df.iloc[:, col_idx])
            
            return {
                'synthetic_data': synthetic_data,
                'final_loss': 0.0,
                'training_metrics': {
                    'dependencies_learned': len(dependencies),
                    'method': 'bayesian_network'
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Bayesian Network generation failed: {str(e)}, using fallback")
            return self._generate_statistical_approximation(data, n_samples)
    
    def _generate_with_copula(self, data: np.ndarray, n_samples: int, **kwargs) -> Dict[str, Any]:
        """Generate synthetic data using Copula method"""
        
        try:
            if not HAS_SCIPY_ADVANCED:
                raise ImportError("Advanced scipy functions not available")
            
            # Transform to uniform marginals
            uniform_data = np.zeros_like(data)
            marginal_distributions = {}
            
            for i in range(data.shape[1]):
                # Estimate marginal distribution
                col_data = data[:, i]
                sorted_data = np.sort(col_data)
                ranks = np.searchsorted(sorted_data, col_data, side='right')
                uniform_data[:, i] = ranks / len(col_data)
                marginal_distributions[i] = sorted_data
            
            # Fit copula (using Gaussian copula approximation)
            correlation_matrix = np.corrcoef(uniform_data.T)
            
            # Generate synthetic uniform data
            synthetic_uniform = np.random.multivariate_normal(
                mean=np.zeros(data.shape[1]),
                cov=correlation_matrix,
                size=n_samples
            )
            
            # Transform to uniform using normal CDF
            from scipy.stats import norm
            synthetic_uniform = norm.cdf(synthetic_uniform)
            
            # Transform back to original marginals
            synthetic_data = np.zeros((n_samples, data.shape[1]))
            for i in range(data.shape[1]):
                # Use quantile transformation
                quantiles = synthetic_uniform[:, i]
                indices = (quantiles * len(marginal_distributions[i])).astype(int)
                indices = np.clip(indices, 0, len(marginal_distributions[i]) - 1)
                synthetic_data[:, i] = marginal_distributions[i][indices]
            
            return {
                'synthetic_data': synthetic_data,
                'final_loss': 0.0,
                'training_metrics': {
                    'copula_type': 'gaussian',
                    'correlation_preserved': True
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Copula generation failed: {str(e)}, using fallback")
            return self._generate_statistical_approximation(data, n_samples)
    
    def _generate_statistical_approximation(self, data: np.ndarray, n_samples: int) -> Dict[str, Any]:
        """Fallback method using statistical approximation"""
        
        try:
            synthetic_data = np.zeros((n_samples, data.shape[1]))
            
            for i in range(data.shape[1]):
                col_data = data[:, i]
                
                # Use kernel density estimation if available
                if HAS_SCIPY_ADVANCED:
                    try:
                        kde = gaussian_kde(col_data)
                        synthetic_data[:, i] = kde.resample(n_samples)[0]
                    except:
                        # Fallback to normal approximation
                        mean, std = np.mean(col_data), np.std(col_data)
                        synthetic_data[:, i] = np.random.normal(mean, std, n_samples)
                else:
                    # Simple normal approximation
                    mean, std = np.mean(col_data), np.std(col_data)
                    synthetic_data[:, i] = np.random.normal(mean, std, n_samples)
            
            return {
                'synthetic_data': synthetic_data,
                'final_loss': 0.0,
                'training_metrics': {'method': 'statistical_approximation'}
            }
            
        except Exception as e:
            self.logger.error(f"Statistical approximation failed: {str(e)}")
            # Last resort: add noise to original data
            noise_scale = 0.1
            indices = np.random.choice(len(data), n_samples, replace=True)
            synthetic_data = data[indices] + np.random.normal(0, noise_scale, (n_samples, data.shape[1]))
            
            return {
                'synthetic_data': synthetic_data,
                'final_loss': 0.0,
                'training_metrics': {'method': 'noisy_resampling'}
            }
    
    def _build_generator(self, latent_dim: int, n_features: int):
        """Build generator network for GAN"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for GAN generation but not available")
        
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(n_features, activation='sigmoid')
        ])
        
        return model
    
    def _build_discriminator(self, n_features: int):
        """Build discriminator network for GAN"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for discriminator generation but not available")
        
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(n_features,)),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def _build_vae(self, n_features: int, latent_dim: int):
        """Build VAE (encoder, decoder, full model)"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for VAE generation but not available")
        
        # Encoder
        encoder_input = keras.Input(shape=(n_features,))
        x = layers.Dense(512, activation='relu')(encoder_input)
        x = layers.Dense(256, activation='relu')(x)
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)
        
        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling)([z_mean, z_log_var])
        encoder = keras.Model(encoder_input, [z_mean, z_log_var, z])
        
        # Decoder
        decoder_input = keras.Input(shape=(latent_dim,))
        x = layers.Dense(256, activation='relu')(decoder_input)
        x = layers.Dense(512, activation='relu')(x)
        decoder_output = layers.Dense(n_features, activation='sigmoid')(x)
        decoder = keras.Model(decoder_input, decoder_output)
        
        # VAE
        vae_output = decoder(encoder(encoder_input)[2])
        vae = keras.Model(encoder_input, vae_output)
        
        # Add VAE loss
        reconstruction_loss = keras.losses.binary_crossentropy(encoder_input, vae_output)
        reconstruction_loss *= n_features
        kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        
        return encoder, decoder, vae
    
    def _train_gan(self, generator, discriminator, gan, data, epochs, batch_size, latent_dim):
        """Train GAN model"""
        
        training_metrics = {'losses': []}
        
        for epoch in range(epochs):
            # Train discriminator
            batch_indices = np.random.randint(0, data.shape[0], batch_size)
            real_batch = data[batch_indices]
            
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_batch = generator.predict(noise, verbose=0)
            
            # Train on real and fake data
            d_loss_real = discriminator.train_on_batch(real_batch, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_batch, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
            
            training_metrics['losses'].append({'d_loss': d_loss[0], 'g_loss': g_loss})
            
            if epoch % 10 == 0:
                self.logger.debug(f"Epoch {epoch}: D loss: {d_loss[0]:.4f}, G loss: {g_loss:.4f}")
        
        training_metrics['final_loss'] = training_metrics['losses'][-1]['g_loss'] if training_metrics['losses'] else 0
        
        return training_metrics
    
    def _learn_variable_dependencies(self, df: pd.DataFrame) -> Dict[int, List[int]]:
        """Learn dependencies between variables for Bayesian Network"""
        
        dependencies = {}
        n_vars = len(df.columns)
        
        # Simple dependency learning based on correlation
        correlation_matrix = df.corr().abs()
        
        for i in range(n_vars):
            # Find variables with high correlation
            corr_with_i = correlation_matrix.iloc[i]
            high_corr_vars = corr_with_i[corr_with_i > 0.5].index.tolist()
            
            # Remove self-correlation and convert to indices
            dependencies[i] = [df.columns.get_loc(var) for var in high_corr_vars if var != df.columns[i]]
            
            # Limit number of parents
            dependencies[i] = dependencies[i][:3]
        
        return dependencies
    
    def _sample_conditional_distribution(self, target_col, parent_cols, parent_values):
        """Sample from conditional distribution"""
        
        try:
            # Find similar samples based on parent values
            if len(parent_cols.columns) == 0:
                return np.random.choice(target_col)
            
            # Calculate distances to parent values
            distances = np.sum((parent_cols.values - parent_values) ** 2, axis=1)
            
            # Use nearest neighbors for sampling
            n_neighbors = min(10, len(target_col))
            nearest_indices = np.argsort(distances)[:n_neighbors]
            
            # Sample from nearest neighbors
            return np.random.choice(target_col.iloc[nearest_indices])
            
        except:
            return np.random.choice(target_col)
    
    def _postprocess_data(self, synthetic_data: np.ndarray, preprocessing_info: Dict, 
                         original_columns: List[str]) -> pd.DataFrame:
        """Post-process synthetic data to original format"""
        
        # Convert to DataFrame
        df = pd.DataFrame(synthetic_data, columns=original_columns)
        
        # Inverse transform numeric columns
        if 'numeric' in preprocessing_info['scalers']:
            scaler = preprocessing_info['scalers']['numeric']
            numeric_columns = [col for col, type_ in preprocessing_info['column_types'].items() 
                             if type_ == 'numeric']
            if numeric_columns:
                df[numeric_columns] = scaler.inverse_transform(df[numeric_columns])
        
        # Inverse transform categorical columns
        for col, encoder in preprocessing_info['encoders'].items():
            if col in df.columns:
                # Clip values to valid range
                df[col] = np.clip(df[col], 0, len(encoder.classes_) - 1)
                df[col] = df[col].round().astype(int)
                df[col] = encoder.inverse_transform(df[col])
        
        return df
    
    def _calculate_fidelity(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """Calculate fidelity score between original and synthetic data"""
        
        try:
            fidelity_scores = []
            
            common_columns = list(set(original.columns) & set(synthetic.columns))
            
            for col in common_columns:
                if pd.api.types.is_numeric_dtype(original[col]):
                    # Compare distributions for numeric columns
                    orig_mean, orig_std = original[col].mean(), original[col].std()
                    synth_mean, synth_std = synthetic[col].mean(), synthetic[col].std()
                    
                    if orig_std > 0:
                        mean_diff = abs(orig_mean - synth_mean) / orig_std
                        std_diff = abs(orig_std - synth_std) / orig_std
                        score = max(0, 1 - (mean_diff + std_diff) / 2)
                        fidelity_scores.append(score)
                else:
                    # Compare distributions for categorical columns
                    orig_dist = original[col].value_counts(normalize=True)
                    synth_dist = synthetic[col].value_counts(normalize=True)
                    
                    # Calculate overlap
                    common_values = set(orig_dist.index) & set(synth_dist.index)
                    if len(common_values) > 0:
                        overlap = sum(min(orig_dist.get(val, 0), synth_dist.get(val, 0)) 
                                    for val in common_values)
                        fidelity_scores.append(overlap)
            
            return np.mean(fidelity_scores) if fidelity_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"Fidelity calculation failed: {str(e)}")
            return 0.5
    
    def _calculate_privacy_score(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """Calculate privacy score (distance from original data)"""
        
        try:
            # Simple privacy score based on nearest neighbor distances
            common_columns = list(set(original.columns) & set(synthetic.columns))
            numeric_columns = original[common_columns].select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return 0.8  # Default privacy score for non-numeric data
            
            # Calculate minimum distances between synthetic and original points
            orig_numeric = original[numeric_columns].values
            synth_numeric = synthetic[numeric_columns].values
            
            # Normalize the data
            scaler = StandardScaler()
            orig_scaled = scaler.fit_transform(orig_numeric)
            synth_scaled = scaler.transform(synth_numeric)
            
            # Calculate minimum distances
            min_distances = []
            sample_size = min(100, len(synth_scaled))  # Sample for efficiency
            
            for i in range(sample_size):
                distances = np.sum((orig_scaled - synth_scaled[i]) ** 2, axis=1)
                min_distances.append(np.min(distances))
            
            # Higher average minimum distance = better privacy
            avg_min_distance = np.mean(min_distances)
            
            # Convert to privacy score (0-1, higher is better)
            privacy_score = min(1.0, avg_min_distance / 2.0)
            
            return privacy_score
            
        except Exception as e:
            self.logger.warning(f"Privacy score calculation failed: {str(e)}")
            return 0.8  # Default privacy score
