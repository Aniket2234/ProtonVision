"""
Report Generator Module for SafeData Pipeline
Generates comprehensive Privacy-Utility assessment reports in multiple formats
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import report generation libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from jinja2 import Template, Environment, FileSystemLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

class ReportGenerator:
    """Comprehensive report generation for privacy-utility analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.report_templates_dir = Path(__file__).parent / "templates"
        self.report_templates_dir.mkdir(exist_ok=True)
        
    def generate_report(self, dataset: pd.DataFrame, report_type: str, 
                       output_path: str, analysis_results: Optional[Dict[str, Any]] = None) -> bool:
        """
        Generate comprehensive privacy-utility report
        
        Args:
            dataset: The dataset being analyzed
            report_type: Type of report ('Executive Summary', 'Technical Report', 'Compliance Report', 'Full Assessment')
            output_path: Path to save the report
            analysis_results: Results from privacy analysis modules
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Generating {report_type} report")
        
        try:
            # Prepare report data
            report_data = self._prepare_report_data(dataset, analysis_results)
            
            # Determine output format from file extension
            output_path = Path(output_path)
            output_format = output_path.suffix.lower()
            
            if output_format == '.pdf':
                success = self._generate_pdf_report(report_data, report_type, output_path)
            elif output_format == '.html':
                success = self._generate_html_report(report_data, report_type, output_path)
            elif output_format == '.json':
                success = self._generate_json_report(report_data, report_type, output_path)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
            if success:
                self.logger.info(f"Report generated successfully: {output_path}")
            else:
                self.logger.error("Report generation failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return False
    
    def _prepare_report_data(self, dataset: pd.DataFrame, 
                           analysis_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare comprehensive data for report generation"""
        
        # Basic dataset information
        dataset_info = {
            'name': 'Dataset Analysis',
            'rows': len(dataset),
            'columns': len(dataset.columns),
            'memory_usage_mb': dataset.memory_usage(deep=True).sum() / (1024 * 1024),
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'column_types': {
                'numeric': len(dataset.select_dtypes(include=[np.number]).columns),
                'categorical': len(dataset.select_dtypes(include=['object', 'category']).columns),
                'datetime': len(dataset.select_dtypes(include=['datetime64']).columns)
            },
            'missing_values': {
                'total': dataset.isnull().sum().sum(),
                'percentage': (dataset.isnull().sum().sum() / (dataset.shape[0] * dataset.shape[1])) * 100
            }
        }
        
        # Privacy analysis results
        privacy_results = analysis_results or {}
        
        # Risk assessment results
        risk_assessment = privacy_results.get('risk_assessment', {})
        risk_summary = {
            'global_risk': risk_assessment.get('global_risk', 0.0),
            'individual_risk_avg': np.mean(risk_assessment.get('individual_risks', [0])),
            'prosecutor_risk': risk_assessment.get('prosecutor_risk', 0.0),
            'journalist_risk': risk_assessment.get('journalist_risk', 0.0),
            'k_anonymity': risk_assessment.get('anonymity_metrics', {}).get('k_anonymity', 0),
            'l_diversity': risk_assessment.get('anonymity_metrics', {}).get('l_diversity', 0),
            'compliance_status': risk_assessment.get('compliance_status', {})
        }
        
        # Privacy enhancement results
        privacy_enhancement = privacy_results.get('privacy_enhancement', {})
        enhancement_summary = {
            'techniques_applied': privacy_enhancement.get('techniques_applied', []),
            'privacy_improvement': privacy_enhancement.get('privacy_improvement', {}),
            'budget_used': privacy_enhancement.get('budget_used', 0.0),
            'enhancement_log': privacy_enhancement.get('enhancement_log', [])
        }
        
        # Utility measurement results
        utility_results = privacy_results.get('utility_measurement', {})
        utility_summary = {
            'overall_utility': utility_results.get('overall_utility', 0.0),
            'mae': utility_results.get('mae', 0.0),
            'rmse': utility_results.get('rmse', 0.0),
            'correlation': utility_results.get('correlation', 0.0),
            'query_accuracy': utility_results.get('query_accuracy', 0.0),
            'model_performance': utility_results.get('model_performance', 0.0),
            'recommendations': utility_results.get('recommendations', [])
        }
        
        # Generate visualizations
        visualizations = self._generate_visualizations(dataset, privacy_results)
        
        # Compile comprehensive report data
        report_data = {
            'metadata': {
                'report_type': 'Privacy-Utility Analysis Report',
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'software': 'SafeData Pipeline v1.0',
                'analyst': 'SafeData System'
            },
            'dataset_info': dataset_info,
            'executive_summary': self._generate_executive_summary(risk_summary, enhancement_summary, utility_summary),
            'risk_assessment': risk_summary,
            'privacy_enhancement': enhancement_summary,
            'utility_analysis': utility_summary,
            'technical_details': self._generate_technical_details(privacy_results),
            'compliance_assessment': self._generate_compliance_assessment(risk_summary),
            'recommendations': self._generate_recommendations(risk_summary, utility_summary),
            'visualizations': visualizations,
            'appendix': self._generate_appendix(dataset, privacy_results)
        }
        
        return report_data
    
    def _generate_executive_summary(self, risk_summary: Dict, enhancement_summary: Dict, 
                                  utility_summary: Dict) -> Dict[str, Any]:
        """Generate executive summary section"""
        
        # Determine overall privacy level
        global_risk = risk_summary.get('global_risk', 0.0)
        if global_risk < 0.01:
            privacy_level = "Excellent"
        elif global_risk < 0.05:
            privacy_level = "Good"
        elif global_risk < 0.1:
            privacy_level = "Moderate"
        else:
            privacy_level = "Poor"
        
        # Determine utility preservation level
        overall_utility = utility_summary.get('overall_utility', 0.0)
        if overall_utility > 0.8:
            utility_level = "Excellent"
        elif overall_utility > 0.6:
            utility_level = "Good"
        elif overall_utility > 0.4:
            utility_level = "Moderate"
        else:
            utility_level = "Poor"
        
        # Key findings
        key_findings = [
            f"Privacy Level: {privacy_level} (Global Risk: {global_risk:.3f})",
            f"Utility Preservation: {utility_level} ({overall_utility:.1%})",
            f"K-Anonymity: {risk_summary.get('k_anonymity', 0)}",
            f"Techniques Applied: {len(enhancement_summary.get('techniques_applied', []))}"
        ]
        
        # Recommendations summary
        top_recommendations = utility_summary.get('recommendations', [])[:3]
        
        return {
            'privacy_level': privacy_level,
            'utility_level': utility_level,
            'key_findings': key_findings,
            'top_recommendations': top_recommendations,
            'dpdp_compliant': risk_summary.get('compliance_status', {}).get('dpdp_compliant', False)
        }
    
    def _generate_technical_details(self, privacy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical details section"""
        
        return {
            'methodology': {
                'risk_assessment_methods': [
                    'Quasi-identifier detection',
                    'Linkage attack simulation',
                    'K-anonymity analysis',
                    'L-diversity assessment'
                ],
                'privacy_techniques': [
                    'Statistical Disclosure Control (SDC)',
                    'Differential Privacy (DP)',
                    'Synthetic Data Generation (SDG)'
                ],
                'utility_metrics': [
                    'Mean Absolute Error (MAE)',
                    'Root Mean Square Error (RMSE)',
                    'Correlation Preservation',
                    'Query Accuracy'
                ]
            },
            'parameters_used': privacy_results.get('parameters', {}),
            'processing_statistics': {
                'total_processing_time': privacy_results.get('processing_time', 0),
                'memory_usage': privacy_results.get('memory_usage', 0),
                'computational_complexity': privacy_results.get('complexity', 'O(n)')
            }
        }
    
    def _generate_compliance_assessment(self, risk_summary: Dict) -> Dict[str, Any]:
        """Generate DPDP Act compliance assessment"""
        
        compliance_status = risk_summary.get('compliance_status', {})
        
        # DPDP Act requirements assessment
        dpdp_requirements = {
            'purpose_limitation': {
                'status': 'Compliant' if compliance_status.get('dpdp_compliant', False) else 'Non-Compliant',
                'description': 'Data processing limited to specified purposes'
            },
            'data_minimization': {
                'status': 'Compliant' if risk_summary.get('global_risk', 1.0) < 0.05 else 'Needs Improvement',
                'description': 'Only necessary data attributes processed'
            },
            'accuracy_maintenance': {
                'status': 'Compliant',  # Assuming accuracy is maintained
                'description': 'Data accuracy preserved during processing'
            },
            'storage_limitation': {
                'status': 'Compliant',
                'description': 'Data retained only as long as necessary'
            }
        }
        
        return {
            'overall_compliance': compliance_status.get('dpdp_compliant', False),
            'dpdp_requirements': dpdp_requirements,
            'risk_level': compliance_status.get('privacy_level', 'Unknown'),
            'compliance_score': self._calculate_compliance_score(dpdp_requirements),
            'next_review_date': (datetime.now().replace(month=datetime.now().month + 6)).strftime('%Y-%m-%d')
        }
    
    def _generate_recommendations(self, risk_summary: Dict, utility_summary: Dict) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Risk-based recommendations
        global_risk = risk_summary.get('global_risk', 0.0)
        if global_risk > 0.1:
            recommendations.append({
                'category': 'Privacy Enhancement',
                'priority': 'High',
                'recommendation': 'Apply additional privacy-preserving techniques to reduce re-identification risk',
                'rationale': f'Current global risk ({global_risk:.3f}) exceeds recommended threshold (0.05)'
            })
        
        if risk_summary.get('k_anonymity', 0) < 3:
            recommendations.append({
                'category': 'Anonymization',
                'priority': 'Medium',
                'recommendation': 'Increase k-anonymity level through generalization or suppression',
                'rationale': 'Current k-anonymity level provides insufficient protection'
            })
        
        # Utility-based recommendations
        overall_utility = utility_summary.get('overall_utility', 0.0)
        if overall_utility < 0.5:
            recommendations.append({
                'category': 'Utility Preservation',
                'priority': 'High',
                'recommendation': 'Adjust privacy parameters to improve data utility',
                'rationale': f'Current utility level ({overall_utility:.1%}) is below acceptable threshold'
            })
        
        # General recommendations
        recommendations.extend([
            {
                'category': 'Monitoring',
                'priority': 'Medium',
                'recommendation': 'Implement continuous privacy monitoring',
                'rationale': 'Regular assessment ensures ongoing compliance'
            },
            {
                'category': 'Documentation',
                'priority': 'Low',
                'recommendation': 'Maintain detailed records of privacy enhancement activities',
                'rationale': 'Required for audit trails and compliance verification'
            }
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _generate_visualizations(self, dataset: pd.DataFrame, privacy_results: Dict) -> Dict[str, str]:
        """Generate visualization charts as base64 encoded images"""
        
        visualizations = {}
        
        try:
            plt.style.use('default')
            
            # Risk distribution histogram
            risk_data = privacy_results.get('risk_assessment', {}).get('individual_risks', [])
            if risk_data:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(risk_data, bins=30, alpha=0.7, color='red', edgecolor='black')
                ax.set_title('Individual Risk Distribution')
                ax.set_xlabel('Re-identification Risk')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Convert to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                visualizations['risk_distribution'] = base64.b64encode(buffer.getvalue()).decode()
                buffer.close()
                plt.close()
            
            # Utility metrics comparison
            utility_data = privacy_results.get('utility_measurement', {})
            if utility_data:
                metrics = ['MAE', 'RMSE', 'Correlation', 'Query Accuracy', 'Model Performance']
                values = [
                    1 - utility_data.get('mae', 0),  # Invert MAE (lower is better)
                    1 - utility_data.get('rmse', 0),  # Invert RMSE (lower is better)
                    utility_data.get('correlation', 0),
                    utility_data.get('query_accuracy', 0),
                    utility_data.get('model_performance', 0)
                ]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(metrics, values, color=['blue', 'green', 'orange', 'red', 'purple'])
                ax.set_title('Utility Preservation Metrics')
                ax.set_ylabel('Score (0-1)')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                visualizations['utility_metrics'] = base64.b64encode(buffer.getvalue()).decode()
                buffer.close()
                plt.close()
            
            # Privacy-Utility Trade-off
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Sample data for trade-off visualization
            privacy_levels = np.linspace(0.01, 0.5, 20)
            utility_levels = 1 - privacy_levels * 2  # Inverse relationship
            
            ax.plot(privacy_levels, utility_levels, 'b-', linewidth=2, label='Trade-off Curve')
            
            # Mark current position
            current_risk = privacy_results.get('risk_assessment', {}).get('global_risk', 0.1)
            current_utility = privacy_results.get('utility_measurement', {}).get('overall_utility', 0.5)
            ax.plot(current_risk, current_utility, 'ro', markersize=10, label='Current Position')
            
            ax.set_xlabel('Privacy Risk')
            ax.set_ylabel('Data Utility')
            ax.set_title('Privacy-Utility Trade-off')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            visualizations['privacy_utility_tradeoff'] = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {str(e)}")
        
        return visualizations
    
    def _generate_appendix(self, dataset: pd.DataFrame, privacy_results: Dict) -> Dict[str, Any]:
        """Generate appendix with detailed technical information"""
        
        return {
            'column_statistics': self._get_column_statistics(dataset),
            'algorithm_details': {
                'risk_assessment': 'Implements linkage attack simulation and k-anonymity analysis',
                'differential_privacy': 'Uses Laplace and Gaussian mechanisms with configurable epsilon',
                'synthetic_data': 'Employs GANs, VAEs, and statistical methods for data generation'
            },
            'performance_metrics': privacy_results.get('performance_metrics', {}),
            'configuration_used': privacy_results.get('configuration', {}),
            'references': [
                'Dwork, C. (2006). Differential Privacy. ICALP 2006.',
                'Sweeney, L. (2002). k-anonymity: A model for protecting privacy. IJUFKS.',
                'Machanavajjhala, A. et al. (2007). l-diversity: Privacy beyond k-anonymity. TKDD.'
            ]
        }
    
    def _get_column_statistics(self, dataset: pd.DataFrame) -> Dict[str, Dict]:
        """Get detailed statistics for each column"""
        
        column_stats = {}
        
        for column in dataset.columns:
            stats = {
                'data_type': str(dataset[column].dtype),
                'non_null_count': int(dataset[column].count()),
                'null_count': int(dataset[column].isnull().sum()),
                'unique_count': int(dataset[column].nunique())
            }
            
            if pd.api.types.is_numeric_dtype(dataset[column]):
                stats.update({
                    'mean': float(dataset[column].mean()),
                    'std': float(dataset[column].std()),
                    'min': float(dataset[column].min()),
                    'max': float(dataset[column].max()),
                    'median': float(dataset[column].median())
                })
            
            column_stats[column] = stats
        
        return column_stats
    
    def _calculate_compliance_score(self, requirements: Dict) -> float:
        """Calculate overall compliance score"""
        
        compliant_count = sum(1 for req in requirements.values() 
                            if req['status'] == 'Compliant')
        total_count = len(requirements)
        
        return compliant_count / total_count if total_count > 0 else 0.0
    
    def _generate_pdf_report(self, report_data: Dict, report_type: str, output_path: Path) -> bool:
        """Generate PDF report using ReportLab"""
        
        if not HAS_REPORTLAB:
            self.logger.error("ReportLab not available for PDF generation")
            return False
        
        try:
            doc = SimpleDocTemplate(str(output_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph(f"SafeData Pipeline - {report_type}", title_style))
            story.append(Spacer(1, 20))
            
            # Metadata
            story.append(Paragraph("Report Information", styles['Heading2']))
            metadata_table = [
                ['Generated:', report_data['metadata']['generated_at']],
                ['Software:', report_data['metadata']['software']],
                ['Dataset Rows:', str(report_data['dataset_info']['rows'])],
                ['Dataset Columns:', str(report_data['dataset_info']['columns'])]
            ]
            
            table = Table(metadata_table, colWidths=[2*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 0), (0, -1), colors.grey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            exec_summary = report_data['executive_summary']
            story.append(Paragraph(f"<b>Privacy Level:</b> {exec_summary['privacy_level']}", styles['Normal']))
            story.append(Paragraph(f"<b>Utility Level:</b> {exec_summary['utility_level']}", styles['Normal']))
            story.append(Paragraph(f"<b>DPDP Compliant:</b> {'Yes' if exec_summary['dpdp_compliant'] else 'No'}", styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Key Findings
            story.append(Paragraph("Key Findings:", styles['Heading3']))
            for finding in exec_summary['key_findings']:
                story.append(Paragraph(f"• {finding}", styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Recommendations
            story.append(Paragraph("Top Recommendations:", styles['Heading3']))
            for rec in exec_summary['top_recommendations']:
                story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Risk Assessment
            story.append(Paragraph("Risk Assessment Summary", styles['Heading2']))
            risk_data = report_data['risk_assessment']
            risk_table = [
                ['Metric', 'Value'],
                ['Global Risk', f"{risk_data['global_risk']:.4f}"],
                ['K-Anonymity', str(risk_data['k_anonymity'])],
                ['L-Diversity', str(risk_data['l_diversity'])],
                ['Prosecutor Risk', f"{risk_data['prosecutor_risk']:.4f}"],
                ['Journalist Risk', f"{risk_data['journalist_risk']:.4f}"]
            ]
            
            table = Table(risk_table, colWidths=[2.5*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Privacy Enhancement
            story.append(Paragraph("Privacy Enhancement Summary", styles['Heading2']))
            enhancement_data = report_data['privacy_enhancement']
            story.append(Paragraph(f"<b>Techniques Applied:</b> {', '.join(enhancement_data['techniques_applied'])}", styles['Normal']))
            story.append(Paragraph(f"<b>Privacy Budget Used:</b> {enhancement_data['budget_used']:.3f}", styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Utility Analysis
            story.append(Paragraph("Utility Analysis Summary", styles['Heading2']))
            utility_data = report_data['utility_analysis']
            utility_table = [
                ['Metric', 'Score'],
                ['Overall Utility', f"{utility_data['overall_utility']:.3f}"],
                ['Mean Absolute Error', f"{utility_data['mae']:.4f}"],
                ['Correlation Preservation', f"{utility_data['correlation']:.3f}"],
                ['Query Accuracy', f"{utility_data['query_accuracy']:.3f}"],
                ['Model Performance', f"{utility_data['model_performance']:.3f}"]
            ]
            
            table = Table(utility_table, colWidths=[3*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            
            # Build PDF
            doc.build(story)
            return True
            
        except Exception as e:
            self.logger.error(f"PDF generation failed: {str(e)}")
            return False
    
    def _generate_html_report(self, report_data: Dict, report_type: str, output_path: Path) -> bool:
        """Generate HTML report using Jinja2 template"""
        
        try:
            # Use built-in template if Jinja2 not available
            if not HAS_JINJA2:
                return self._generate_simple_html_report(report_data, report_type, output_path)
            
            # Load template
            template_path = self.report_templates_dir / "report_template.html"
            if not template_path.exists():
                # Create template if it doesn't exist
                self._create_html_template(template_path)
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            template = Template(template_content)
            
            # Render template
            html_content = template.render(
                report_type=report_type,
                report_data=report_data,
                generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Save HTML file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return True
            
        except Exception as e:
            self.logger.error(f"HTML generation failed: {str(e)}")
            return False
    
    def _generate_simple_html_report(self, report_data: Dict, report_type: str, output_path: Path) -> bool:
        """Generate simple HTML report without Jinja2"""
        
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SafeData Pipeline - {report_type}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 3px solid #2c3e50; padding-bottom: 20px; margin-bottom: 30px; }}
        .header h1 {{ color: #2c3e50; margin: 0; font-size: 2.5em; }}
        .header p {{ color: #7f8c8d; margin: 10px 0 0 0; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #3498db; }}
        .metric-card h3 {{ margin: 0 0 10px 0; color: #2c3e50; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #27ae60; }}
        .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .table th {{ background-color: #34495e; color: white; }}
        .recommendations {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 6px; padding: 20px; }}
        .recommendation-item {{ margin-bottom: 15px; padding: 10px; background: white; border-radius: 4px; }}
        .priority-high {{ border-left: 4px solid #e74c3c; }}
        .priority-medium {{ border-left: 4px solid #f39c12; }}
        .priority-low {{ border-left: 4px solid #27ae60; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SafeData Pipeline Report</h1>
            <p>{report_type} • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Privacy Level</h3>
                    <div class="metric-value">{report_data['executive_summary']['privacy_level']}</div>
                </div>
                <div class="metric-card">
                    <h3>Utility Level</h3>
                    <div class="metric-value">{report_data['executive_summary']['utility_level']}</div>
                </div>
                <div class="metric-card">
                    <h3>DPDP Compliant</h3>
                    <div class="metric-value">{'Yes' if report_data['executive_summary']['dpdp_compliant'] else 'No'}</div>
                </div>
                <div class="metric-card">
                    <h3>Dataset Size</h3>
                    <div class="metric-value">{report_data['dataset_info']['rows']:,} rows</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Risk Assessment</h2>
            <table class="table">
                <thead>
                    <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Global Risk</td>
                        <td>{report_data['risk_assessment']['global_risk']:.4f}</td>
                        <td>{'✓ Low' if report_data['risk_assessment']['global_risk'] < 0.05 else '⚠ High'}</td>
                    </tr>
                    <tr>
                        <td>K-Anonymity</td>
                        <td>{report_data['risk_assessment']['k_anonymity']}</td>
                        <td>{'✓ Good' if report_data['risk_assessment']['k_anonymity'] >= 3 else '⚠ Low'}</td>
                    </tr>
                    <tr>
                        <td>L-Diversity</td>
                        <td>{report_data['risk_assessment']['l_diversity']}</td>
                        <td>{'✓ Good' if report_data['risk_assessment']['l_diversity'] >= 2 else '⚠ Low'}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>Privacy Enhancement</h2>
            <p><strong>Techniques Applied:</strong> {', '.join(report_data['privacy_enhancement']['techniques_applied']) if report_data['privacy_enhancement']['techniques_applied'] else 'None'}</p>
            <p><strong>Privacy Budget Used:</strong> {report_data['privacy_enhancement']['budget_used']:.3f}</p>
        </div>
        
        <div class="section">
            <h2>Utility Analysis</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Overall Utility</h3>
                    <div class="metric-value">{report_data['utility_analysis']['overall_utility']:.1%}</div>
                </div>
                <div class="metric-card">
                    <h3>Query Accuracy</h3>
                    <div class="metric-value">{report_data['utility_analysis']['query_accuracy']:.1%}</div>
                </div>
                <div class="metric-card">
                    <h3>Model Performance</h3>
                    <div class="metric-value">{report_data['utility_analysis']['model_performance']:.1%}</div>
                </div>
                <div class="metric-card">
                    <h3>Correlation</h3>
                    <div class="metric-value">{report_data['utility_analysis']['correlation']:.3f}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <div class="recommendations">
"""
            
            for rec in report_data['recommendations'][:5]:
                priority_class = f"priority-{rec.get('priority', 'medium').lower()}"
                html_content += f"""
                <div class="recommendation-item {priority_class}">
                    <strong>{rec.get('category', 'General')}</strong> ({rec.get('priority', 'Medium')} Priority)<br>
                    {rec.get('recommendation', 'No recommendation available')}<br>
                    <small><em>{rec.get('rationale', '')}</em></small>
                </div>
"""
            
            html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>Technical Details</h2>
            <p><strong>Processing Time:</strong> N/A</p>
            <p><strong>Memory Usage:</strong> {:.2f} MB</p>
            <p><strong>Algorithm Complexity:</strong> O(n)</p>
        </div>
    </div>
</body>
</html>
""".format(report_data['dataset_info']['memory_usage_mb'])
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Simple HTML generation failed: {str(e)}")
            return False
    
    def _generate_json_report(self, report_data: Dict, report_type: str, output_path: Path) -> bool:
        """Generate JSON report"""
        
        try:
            # Add report metadata
            json_report = {
                'report_metadata': {
                    'type': report_type,
                    'generated_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'format': 'json'
                },
                'data': report_data
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, ensure_ascii=False, default=str)
            
            return True
            
        except Exception as e:
            self.logger.error(f"JSON generation failed: {str(e)}")
            return False
    
    def _create_html_template(self, template_path: Path):
        """Create default HTML template"""
        
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SafeData Pipeline - {{ report_type }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 3px solid #2c3e50; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { color: #2c3e50; margin: 0; font-size: 2.5em; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #3498db; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #27ae60; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SafeData Pipeline Report</h1>
            <p>{{ report_type }} • Generated: {{ generated_at }}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Privacy Level</h3>
                    <div class="metric-value">{{ report_data.executive_summary.privacy_level }}</div>
                </div>
                <div class="metric-card">
                    <h3>Utility Level</h3>
                    <div class="metric-value">{{ report_data.executive_summary.utility_level }}</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
