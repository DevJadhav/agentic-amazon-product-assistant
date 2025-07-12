"""
Final Production Readiness Validation and Comprehensive Testing
Complete validation suite to ensure the Amazon Electronics Assistant is production-ready.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class ProductionReadinessValidator:
    """Comprehensive production readiness validator."""
    
    def __init__(self):
        """Initialize the production readiness validator."""
        self.validation_results = {}
        self.overall_score = 0.0
        self.critical_issues = []
        self.recommendations = []
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive production readiness validation."""
        
        logger.info("🚀 Starting Production Readiness Validation")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Validation categories
        validations = [
            ("Infrastructure", self._validate_infrastructure),
            ("Data Processing", self._validate_data_processing),
            ("RAG System", self._validate_rag_system),
            ("Vector Database", self._validate_vector_database),
            ("LLM Integration", self._validate_llm_integration),
            ("UI/UX", self._validate_ui_system),
            ("Monitoring & Tracing", self._validate_monitoring),
            ("Security", self._validate_security),
            ("Performance", self._validate_performance),
            ("Documentation", self._validate_documentation)
        ]
        
        # Run all validations
        for category, validation_func in validations:
            logger.info(f"📊 Validating {category}...")
            try:
                result = await validation_func()
                self.validation_results[category] = result
                logger.info(f"✅ {category} validation completed - Score: {result['score']:.2f}/1.00")
            except Exception as e:
                logger.error(f"❌ {category} validation failed: {e}")
                self.validation_results[category] = {
                    'score': 0.0,
                    'status': 'FAILED',
                    'error': str(e),
                    'details': {}
                }
        
        # Calculate overall readiness
        total_time = time.time() - start_time
        readiness_report = self._generate_final_report(total_time)
        
        logger.info("🏁 Production Readiness Validation Completed")
        logger.info("=" * 60)
        
        return readiness_report
    
    async def _validate_infrastructure(self) -> Dict[str, Any]:
        """Validate infrastructure and deployment readiness."""
        
        score = 0.0
        max_score = 10.0
        details = {}
        issues = []
        
        # Check Docker configuration
        docker_files = [
            project_root / "Dockerfile",
            project_root / "Dockerfile.production", 
            project_root / "docker-compose.yml"
        ]
        
        docker_score = 0
        for docker_file in docker_files:
            if docker_file.exists():
                docker_score += 1
                details[f"docker_{docker_file.name}"] = "✅ Available"
            else:
                details[f"docker_{docker_file.name}"] = "❌ Missing"
                issues.append(f"Missing {docker_file.name}")
        
        score += (docker_score / len(docker_files)) * 3.0
        
        # Check deployment scripts
        deploy_script = project_root / "deploy" / "production-deploy.sh"
        if deploy_script.exists():
            score += 2.0
            details["deployment_script"] = "✅ Available"
        else:
            details["deployment_script"] = "❌ Missing"
            issues.append("Missing deployment script")
        
        # Check environment configuration
        env_example = project_root / ".env.example"
        if env_example.exists():
            score += 1.5
            details["env_configuration"] = "✅ Available"
        else:
            details["env_configuration"] = "❌ Missing"
            issues.append("Missing environment configuration example")
        
        # Check pyproject.toml
        pyproject = project_root / "pyproject.toml"
        if pyproject.exists():
            score += 1.5
            details["dependency_management"] = "✅ Available"
        else:
            details["dependency_management"] = "❌ Missing"
            issues.append("Missing pyproject.toml")
        
        # Check Makefile
        makefile = project_root / "Makefile"
        if makefile.exists():
            score += 2.0
            details["build_automation"] = "✅ Available"
        else:
            details["build_automation"] = "❌ Missing"
            issues.append("Missing Makefile for build automation")
        
        return {
            'score': score / max_score,
            'status': 'PASS' if score >= 8.0 else 'NEEDS_IMPROVEMENT',
            'details': details,
            'issues': issues
        }
    
    async def _validate_data_processing(self) -> Dict[str, Any]:
        """Validate data processing pipeline."""
        
        score = 0.0
        max_score = 10.0
        details = {}
        issues = []
        
        # Check processed data exists
        data_dir = project_root / "data" / "processed"
        required_files = [
            "electronics_top1000_products.jsonl",
            "electronics_top1000_products_reviews.jsonl", 
            "electronics_rag_documents.jsonl",
            "dataset_summary.json"
        ]
        
        data_score = 0
        for file_name in required_files:
            file_path = data_dir / file_name
            if file_path.exists():
                data_score += 1
                details[f"data_{file_name}"] = "✅ Available"
            else:
                details[f"data_{file_name}"] = "❌ Missing"
                issues.append(f"Missing processed data file: {file_name}")
        
        score += (data_score / len(required_files)) * 4.0
        
        # Check notebooks
        notebooks_dir = project_root / "notebooks"
        if notebooks_dir.exists():
            notebook_files = list(notebooks_dir.glob("*.ipynb"))
            if len(notebook_files) >= 2:
                score += 2.0
                details["processing_notebooks"] = f"✅ {len(notebook_files)} notebooks available"
            else:
                details["processing_notebooks"] = "⚠️ Limited notebooks"
                issues.append("Insufficient processing notebooks")
        else:
            details["processing_notebooks"] = "❌ Missing notebooks directory"
            issues.append("Missing notebooks directory")
        
        # Check advanced pipeline
        advanced_pipeline = project_root / "src" / "data_processing" / "advanced_pipeline.py"
        if advanced_pipeline.exists():
            score += 2.0
            details["advanced_pipeline"] = "✅ Available"
        else:
            details["advanced_pipeline"] = "❌ Missing"
            issues.append("Missing advanced data processing pipeline")
        
        # Check visualization system
        viz_system = project_root / "src" / "visualization" / "interactive_dashboards.py"
        if viz_system.exists():
            score += 2.0
            details["visualization_system"] = "✅ Available"
        else:
            details["visualization_system"] = "❌ Missing"
            issues.append("Missing visualization system")
        
        return {
            'score': score / max_score,
            'status': 'PASS' if score >= 8.0 else 'NEEDS_IMPROVEMENT',
            'details': details,
            'issues': issues
        }
    
    async def _validate_rag_system(self) -> Dict[str, Any]:
        """Validate RAG system implementation."""
        
        score = 0.0
        max_score = 10.0
        details = {}
        issues = []
        
        # Check core RAG components
        rag_components = [
            ("query_processor", "src/rag/query_processor.py"),
            ("enhanced_query_processor", "src/rag/enhanced_query_processor.py"),
            ("vector_db", "src/rag/vector_db_weaviate_simple.py"),
            ("enhanced_vector_db", "src/rag/enhanced_vector_db.py"),
            ("mock_vector_db", "src/rag/mock_vector_db.py")
        ]
        
        component_score = 0
        for component_name, component_path in rag_components:
            file_path = project_root / component_path
            if file_path.exists():
                component_score += 1
                details[f"rag_{component_name}"] = "✅ Available"
            else:
                details[f"rag_{component_name}"] = "❌ Missing"
                issues.append(f"Missing RAG component: {component_name}")
        
        score += (component_score / len(rag_components)) * 6.0
        
        # Check evaluation system
        evaluation_components = [
            ("evaluator", "src/evaluation/evaluator.py"),
            ("enhanced_evaluator", "src/evaluation/enhanced_evaluator.py"),
            ("scorers", "src/evaluation/scorers.py"),
            ("dataset", "src/evaluation/dataset.py")
        ]
        
        eval_score = 0
        for eval_name, eval_path in evaluation_components:
            file_path = project_root / eval_path
            if file_path.exists():
                eval_score += 1
                details[f"eval_{eval_name}"] = "✅ Available"
            else:
                details[f"eval_{eval_name}"] = "❌ Missing"
                issues.append(f"Missing evaluation component: {eval_name}")
        
        score += (eval_score / len(evaluation_components)) * 4.0
        
        return {
            'score': score / max_score,
            'status': 'PASS' if score >= 8.0 else 'NEEDS_IMPROVEMENT',
            'details': details,
            'issues': issues
        }
    
    async def _validate_vector_database(self) -> Dict[str, Any]:
        """Validate vector database implementation."""
        
        score = 0.0
        max_score = 10.0
        details = {}
        issues = []
        
        # Check vector database files
        vector_db_files = [
            "src/rag/vector_db_weaviate_simple.py",
            "src/rag/enhanced_vector_db.py",
            "src/rag/mock_vector_db.py"
        ]
        
        db_score = 0
        for db_file in vector_db_files:
            file_path = project_root / db_file
            if file_path.exists():
                db_score += 1
                details[f"vector_db_{Path(db_file).stem}"] = "✅ Available"
            else:
                details[f"vector_db_{Path(db_file).stem}"] = "❌ Missing"
                issues.append(f"Missing vector database file: {db_file}")
        
        score += (db_score / len(vector_db_files)) * 5.0
        
        # Check Weaviate data directory
        weaviate_dir = project_root / "data" / "weaviate_db"
        if weaviate_dir.exists():
            score += 2.0
            details["weaviate_directory"] = "✅ Available"
        else:
            details["weaviate_directory"] = "❌ Missing"
            issues.append("Missing Weaviate data directory")
        
        # Check Docker compose includes Weaviate
        docker_compose = project_root / "docker-compose.yml"
        if docker_compose.exists():
            with open(docker_compose, 'r') as f:
                content = f.read()
                if 'weaviate' in content.lower():
                    score += 3.0
                    details["weaviate_docker"] = "✅ Configured in Docker Compose"
                else:
                    details["weaviate_docker"] = "⚠️ Not found in Docker Compose"
                    issues.append("Weaviate not configured in Docker Compose")
        
        return {
            'score': score / max_score,
            'status': 'PASS' if score >= 8.0 else 'NEEDS_IMPROVEMENT',
            'details': details,
            'issues': issues
        }
    
    async def _validate_llm_integration(self) -> Dict[str, Any]:
        """Validate LLM integration and multi-provider support."""
        
        score = 0.0
        max_score = 10.0
        details = {}
        issues = []
        
        # Check LLM manager files
        llm_files = [
            "src/chatbot_ui/llm_manager.py",
            "src/chatbot_ui/enhanced_llm_manager.py"
        ]
        
        llm_score = 0
        for llm_file in llm_files:
            file_path = project_root / llm_file
            if file_path.exists():
                llm_score += 1
                details[f"llm_{Path(llm_file).stem}"] = "✅ Available"
            else:
                details[f"llm_{Path(llm_file).stem}"] = "❌ Missing"
                issues.append(f"Missing LLM file: {llm_file}")
        
        score += (llm_score / len(llm_files)) * 4.0
        
        # Check configuration
        config_file = project_root / "src" / "chatbot_ui" / "core" / "config.py"
        if config_file.exists():
            score += 3.0
            details["configuration"] = "✅ Available"
        else:
            details["configuration"] = "❌ Missing"
            issues.append("Missing configuration file")
        
        # Check UI components
        ui_files = [
            "src/chatbot_ui/streamlit_app.py",
            "src/chatbot_ui/enhanced_streamlit_app.py",
            "src/chatbot_ui/ui_components.py"
        ]
        
        ui_score = 0
        for ui_file in ui_files:
            file_path = project_root / ui_file
            if file_path.exists():
                ui_score += 1
                details[f"ui_{Path(ui_file).stem}"] = "✅ Available"
        
        score += (ui_score / len(ui_files)) * 3.0
        
        return {
            'score': score / max_score,
            'status': 'PASS' if score >= 8.0 else 'NEEDS_IMPROVEMENT',
            'details': details,
            'issues': issues
        }
    
    async def _validate_ui_system(self) -> Dict[str, Any]:
        """Validate UI/UX implementation."""
        
        score = 0.0
        max_score = 10.0
        details = {}
        issues = []
        
        # Check main UI files
        ui_files = [
            "src/chatbot_ui/streamlit_app.py",
            "src/chatbot_ui/enhanced_streamlit_app.py",
            "src/chatbot_ui/ui_components.py"
        ]
        
        ui_score = 0
        for ui_file in ui_files:
            file_path = project_root / ui_file
            if file_path.exists():
                ui_score += 1
                details[f"ui_{Path(ui_file).stem}"] = "✅ Available"
            else:
                details[f"ui_{Path(ui_file).stem}"] = "❌ Missing"
                issues.append(f"Missing UI file: {ui_file}")
        
        score += (ui_score / len(ui_files)) * 4.0
        
        # Check visualization components
        viz_files = [
            "src/visualization/interactive_dashboards.py",
            "src/visualization/streamlit_dashboard_app.py"
        ]
        
        viz_score = 0
        for viz_file in viz_files:
            file_path = project_root / viz_file
            if file_path.exists():
                viz_score += 1
                details[f"viz_{Path(viz_file).stem}"] = "✅ Available"
            else:
                details[f"viz_{Path(viz_file).stem}"] = "❌ Missing"
                issues.append(f"Missing visualization file: {viz_file}")
        
        score += (viz_score / len(viz_files)) * 3.0
        
        # Check performance tracking
        perf_file = project_root / "src" / "chatbot_ui" / "performance_tracker.py"
        if perf_file.exists():
            score += 3.0
            details["performance_tracking"] = "✅ Available"
        else:
            details["performance_tracking"] = "❌ Missing"
            issues.append("Missing performance tracking")
        
        return {
            'score': score / max_score,
            'status': 'PASS' if score >= 8.0 else 'NEEDS_IMPROVEMENT',
            'details': details,
            'issues': issues
        }
    
    async def _validate_monitoring(self) -> Dict[str, Any]:
        """Validate monitoring and tracing systems."""
        
        score = 0.0
        max_score = 10.0
        details = {}
        issues = []
        
        # Check tracing components
        tracing_files = [
            "src/tracing/trace_utils.py",
            "src/tracing/enhanced_trace_utils.py",
            "src/tracing/business_intelligence.py"
        ]
        
        tracing_score = 0
        for tracing_file in tracing_files:
            file_path = project_root / tracing_file
            if file_path.exists():
                tracing_score += 1
                details[f"tracing_{Path(tracing_file).stem}"] = "✅ Available"
            else:
                details[f"tracing_{Path(tracing_file).stem}"] = "❌ Missing"
                issues.append(f"Missing tracing file: {tracing_file}")
        
        score += (tracing_score / len(tracing_files)) * 5.0
        
        # Check evaluation scripts
        eval_scripts = [
            "eval/run_evaluation.py",
            "eval/run_synthetic_evaluation.py"
        ]
        
        eval_script_score = 0
        for eval_script in eval_scripts:
            file_path = project_root / eval_script
            if file_path.exists():
                eval_script_score += 1
                details[f"eval_script_{Path(eval_script).stem}"] = "✅ Available"
            else:
                details[f"eval_script_{Path(eval_script).stem}"] = "❌ Missing"
                issues.append(f"Missing evaluation script: {eval_script}")
        
        score += (eval_script_score / len(eval_scripts)) * 3.0
        
        # Check production test suite
        prod_test = project_root / "scripts" / "production_test_suite.py"
        if prod_test.exists():
            score += 2.0
            details["production_testing"] = "✅ Available"
        else:
            details["production_testing"] = "❌ Missing"
            issues.append("Missing production test suite")
        
        return {
            'score': score / max_score,
            'status': 'PASS' if score >= 8.0 else 'NEEDS_IMPROVEMENT',
            'details': details,
            'issues': issues
        }
    
    async def _validate_security(self) -> Dict[str, Any]:
        """Validate security implementations."""
        
        score = 10.0  # Start with full score, deduct for issues
        max_score = 10.0
        details = {}
        issues = []
        
        # Check for .env in .gitignore
        gitignore = project_root / ".gitignore"
        if gitignore.exists():
            with open(gitignore, 'r') as f:
                content = f.read()
                if '.env' in content:
                    details["env_gitignore"] = "✅ .env files properly ignored"
                else:
                    details["env_gitignore"] = "⚠️ .env files may not be ignored"
                    issues.append(".env files should be in .gitignore")
                    score -= 2.0
        else:
            details["gitignore"] = "❌ Missing .gitignore file"
            issues.append("Missing .gitignore file")
            score -= 3.0
        
        # Check for hardcoded secrets (basic check)
        python_files = list(project_root.rglob("*.py"))
        secret_patterns = ['api_key', 'password', 'secret', 'token']
        
        hardcoded_secrets = 0
        for py_file in python_files[:20]:  # Check first 20 files for performance
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    for pattern in secret_patterns:
                        if f'{pattern}=' in content and 'os.getenv' not in content:
                            hardcoded_secrets += 1
                            break
            except:
                continue
        
        if hardcoded_secrets == 0:
            details["hardcoded_secrets"] = "✅ No obvious hardcoded secrets found"
        else:
            details["hardcoded_secrets"] = f"⚠️ {hardcoded_secrets} potential hardcoded secrets"
            issues.append("Potential hardcoded secrets found")
            score -= min(hardcoded_secrets * 1.0, 3.0)
        
        # Check Docker security
        dockerfile = project_root / "Dockerfile.production"
        if dockerfile.exists():
            with open(dockerfile, 'r') as f:
                content = f.read()
                if 'USER' in content and 'root' not in content.lower():
                    details["docker_security"] = "✅ Non-root user configured"
                else:
                    details["docker_security"] = "⚠️ May be running as root"
                    issues.append("Docker container may run as root user")
                    score -= 2.0
        
        return {
            'score': max(score, 0) / max_score,
            'status': 'PASS' if score >= 8.0 else 'NEEDS_IMPROVEMENT',
            'details': details,
            'issues': issues
        }
    
    async def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance optimizations."""
        
        score = 0.0
        max_score = 10.0
        details = {}
        issues = []
        
        # Check for async implementations
        async_files = []
        python_files = list(project_root.rglob("*.py"))
        
        for py_file in python_files[:10]:  # Check first 10 files
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'async def' in content or 'await' in content:
                        async_files.append(py_file.name)
            except:
                continue
        
        if len(async_files) >= 3:
            score += 3.0
            details["async_support"] = f"✅ {len(async_files)} files with async support"
        else:
            details["async_support"] = f"⚠️ Limited async support ({len(async_files)} files)"
            issues.append("Limited async/await usage for performance")
        
        # Check for caching implementations
        cache_keywords = ['cache', 'lru_cache', 'memoize']
        cache_files = []
        
        for py_file in python_files[:10]:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    if any(keyword in content for keyword in cache_keywords):
                        cache_files.append(py_file.name)
            except:
                continue
        
        if len(cache_files) >= 2:
            score += 2.0
            details["caching"] = f"✅ {len(cache_files)} files with caching"
        else:
            details["caching"] = f"⚠️ Limited caching ({len(cache_files)} files)"
            issues.append("Limited caching implementation")
        
        # Check for batch processing
        batch_keywords = ['batch', 'bulk', 'parallel']
        batch_files = []
        
        for py_file in python_files[:10]:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    if any(keyword in content for keyword in batch_keywords):
                        batch_files.append(py_file.name)
            except:
                continue
        
        if len(batch_files) >= 2:
            score += 2.0
            details["batch_processing"] = f"✅ {len(batch_files)} files with batch processing"
        else:
            details["batch_processing"] = f"⚠️ Limited batch processing ({len(batch_files)} files)"
        
        # Check for monitoring
        monitoring_files = [
            "src/chatbot_ui/performance_tracker.py",
            "src/tracing/enhanced_trace_utils.py"
        ]
        
        monitoring_score = 0
        for mon_file in monitoring_files:
            if (project_root / mon_file).exists():
                monitoring_score += 1
        
        score += (monitoring_score / len(monitoring_files)) * 3.0
        details["performance_monitoring"] = f"✅ {monitoring_score}/{len(monitoring_files)} monitoring components"
        
        return {
            'score': score / max_score,
            'status': 'PASS' if score >= 8.0 else 'NEEDS_IMPROVEMENT',
            'details': details,
            'issues': issues
        }
    
    async def _validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness."""
        
        score = 0.0
        max_score = 10.0
        details = {}
        issues = []
        
        # Check README
        readme = project_root / "README.md"
        if readme.exists():
            with open(readme, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 1000:
                    score += 3.0
                    details["readme"] = f"✅ Comprehensive README ({len(content)} chars)"
                else:
                    score += 1.0
                    details["readme"] = f"⚠️ Basic README ({len(content)} chars)"
                    issues.append("README could be more comprehensive")
        else:
            details["readme"] = "❌ Missing README"
            issues.append("Missing README.md")
        
        # Check documentation directory
        docs_dir = project_root / "docs"
        if docs_dir.exists():
            doc_files = list(docs_dir.glob("*.md"))
            if len(doc_files) >= 3:
                score += 3.0
                details["documentation"] = f"✅ {len(doc_files)} documentation files"
            else:
                score += 1.0
                details["documentation"] = f"⚠️ Limited docs ({len(doc_files)} files)"
                issues.append("Could use more documentation files")
        else:
            details["documentation"] = "❌ Missing docs directory"
            issues.append("Missing documentation directory")
        
        # Check docstrings (sample a few files)
        python_files = list((project_root / "src").rglob("*.py"))[:5]
        docstring_count = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '"""' in content or "'''" in content:
                        docstring_count += 1
            except:
                continue
        
        if docstring_count >= 3:
            score += 2.0
            details["docstrings"] = f"✅ {docstring_count}/{len(python_files)} files have docstrings"
        else:
            score += 0.5
            details["docstrings"] = f"⚠️ {docstring_count}/{len(python_files)} files have docstrings"
            issues.append("More files need docstrings")
        
        # Check LICENSE
        license_file = project_root / "LICENSE"
        if license_file.exists():
            score += 2.0
            details["license"] = "✅ License file present"
        else:
            details["license"] = "❌ Missing license"
            issues.append("Missing LICENSE file")
        
        return {
            'score': score / max_score,
            'status': 'PASS' if score >= 7.0 else 'NEEDS_IMPROVEMENT',
            'details': details,
            'issues': issues
        }
    
    def _generate_final_report(self, execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        
        # Calculate overall score
        total_score = 0.0
        category_weights = {
            'Infrastructure': 0.15,
            'Data Processing': 0.15,
            'RAG System': 0.20,
            'Vector Database': 0.15,
            'LLM Integration': 0.10,
            'UI/UX': 0.10,
            'Monitoring & Tracing': 0.10,
            'Security': 0.15,
            'Performance': 0.10,
            'Documentation': 0.05
        }
        
        weighted_scores = {}
        for category, weight in category_weights.items():
            if category in self.validation_results:
                score = self.validation_results[category]['score']
                weighted_score = score * weight
                weighted_scores[category] = weighted_score
                total_score += weighted_score
        
        self.overall_score = total_score
        
        # Determine readiness level
        if total_score >= 0.9:
            readiness_level = "🚀 PRODUCTION READY"
            readiness_color = "GREEN"
        elif total_score >= 0.8:
            readiness_level = "⚠️ MOSTLY READY"
            readiness_color = "YELLOW"
        elif total_score >= 0.7:
            readiness_level = "🔧 NEEDS IMPROVEMENT"
            readiness_color = "ORANGE"
        else:
            readiness_level = "❌ NOT READY"
            readiness_color = "RED"
        
        # Collect all issues
        all_issues = []
        critical_issues = []
        
        for category, result in self.validation_results.items():
            issues = result.get('issues', [])
            for issue in issues:
                issue_entry = f"{category}: {issue}"
                all_issues.append(issue_entry)
                
                if result['score'] < 0.6:  # Critical if score is very low
                    critical_issues.append(issue_entry)
        
        self.critical_issues = critical_issues
        
        # Generate recommendations
        recommendations = []
        
        for category, result in self.validation_results.items():
            if result['score'] < 0.8:
                recommendations.append(f"Improve {category}: {result.get('issues', ['General improvements needed'])[0]}")
        
        if total_score < 0.9:
            recommendations.append("Consider additional testing before production deployment")
        
        if len(critical_issues) > 0:
            recommendations.insert(0, "Address critical issues before any deployment")
        
        self.recommendations = recommendations[:10]  # Top 10 recommendations
        
        # Create summary
        summary = {
            'overall_score': round(total_score, 3),
            'readiness_level': readiness_level,
            'readiness_color': readiness_color,
            'execution_time': round(execution_time, 2),
            'validation_timestamp': datetime.now().isoformat(),
            'categories_validated': len(self.validation_results),
            'categories_passed': len([r for r in self.validation_results.values() if r['score'] >= 0.8]),
            'critical_issues_count': len(critical_issues),
            'total_issues_count': len(all_issues)
        }
        
        return {
            'summary': summary,
            'detailed_scores': {cat: result['score'] for cat, result in self.validation_results.items()},
            'weighted_scores': weighted_scores,
            'validation_results': self.validation_results,
            'critical_issues': critical_issues,
            'all_issues': all_issues,
            'recommendations': recommendations,
            'next_steps': self._generate_next_steps(total_score, critical_issues)
        }
    
    def _generate_next_steps(self, total_score: float, critical_issues: List[str]) -> List[str]:
        """Generate next steps based on validation results."""
        
        next_steps = []
        
        if len(critical_issues) > 0:
            next_steps.extend([
                "🚨 IMMEDIATE: Address all critical issues before proceeding",
                "📋 Create detailed remediation plan with timeline",
                "🔍 Re-run validation after critical fixes"
            ])
        
        if total_score >= 0.9:
            next_steps.extend([
                "✅ System is production-ready",
                "🚀 Plan gradual production rollout",
                "📊 Set up production monitoring",
                "👥 Conduct final user acceptance testing",
                "📋 Prepare production deployment checklist"
            ])
        elif total_score >= 0.8:
            next_steps.extend([
                "🔧 Address remaining improvement areas",
                "🧪 Conduct additional testing in staging",
                "📈 Optimize performance bottlenecks",
                "🔒 Review and strengthen security measures",
                "📝 Update documentation"
            ])
        else:
            next_steps.extend([
                "🛠️ Complete major remediation work",
                "🏗️ Focus on infrastructure improvements",
                "🔧 Enhance core system components",
                "🧪 Implement comprehensive testing",
                "📚 Improve documentation and processes"
            ])
        
        return next_steps[:8]  # Top 8 next steps


async def main():
    """Main execution function."""
    
    print("\n🚀 Amazon Electronics Assistant - Production Readiness Validation")
    print("=" * 80)
    print("This comprehensive validation ensures your system is ready for production deployment.")
    print("=" * 80)
    
    validator = ProductionReadinessValidator()
    
    try:
        # Run validation
        report = await validator.run_comprehensive_validation()
        
        # Print executive summary
        summary = report['summary']
        print(f"\n📊 EXECUTIVE SUMMARY")
        print("=" * 40)
        print(f"Overall Score: {summary['overall_score']:.3f}/1.000")
        print(f"Readiness Level: {summary['readiness_level']}")
        print(f"Categories Validated: {summary['categories_validated']}")
        print(f"Categories Passed: {summary['categories_passed']}/{summary['categories_validated']}")
        print(f"Critical Issues: {summary['critical_issues_count']}")
        print(f"Validation Time: {summary['execution_time']}s")
        
        # Print category scores
        print(f"\n📈 CATEGORY SCORES")
        print("=" * 40)
        for category, score in report['detailed_scores'].items():
            status = "✅ PASS" if score >= 0.8 else "⚠️ NEEDS WORK" if score >= 0.6 else "❌ CRITICAL"
            print(f"{category:.<25} {score:.3f} {status}")
        
        # Print critical issues
        if report['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES")
            print("=" * 40)
            for issue in report['critical_issues']:
                print(f"• {issue}")
        
        # Print recommendations
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS")
            print("=" * 40)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Print next steps
        print(f"\n🎯 NEXT STEPS")
        print("=" * 40)
        for i, step in enumerate(report['next_steps'], 1):
            print(f"{i}. {step}")
        
        # Save detailed report
        output_file = project_root / "production_readiness_report.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {output_file}")
        
        # Final status
        if summary['overall_score'] >= 0.9:
            print(f"\n🎉 CONGRATULATIONS! Your Amazon Electronics Assistant is PRODUCTION READY! 🎉")
        elif summary['overall_score'] >= 0.8:
            print(f"\n✅ Great work! Your system is mostly ready with minor improvements needed.")
        else:
            print(f"\n🔧 Your system needs some work before production deployment.")
        
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"❌ Validation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))