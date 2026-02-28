"""
Unit tests for Docker Compose configuration validation.

These tests verify that the docker-compose.yml file is valid
and contains all required configurations.
"""

import os
import subprocess
import unittest
from pathlib import Path


class TestDockerCompose(unittest.TestCase):
    """Test suite for Docker Compose configuration."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.project_root = Path(__file__).parent
        cls.compose_file = cls.project_root / "docker-compose.yml"
        cls.dockerfile = cls.project_root / "Dockerfile"

    def test_docker_compose_file_exists(self):
        """Verify docker-compose.yml file exists."""
        self.assertTrue(
            self.compose_file.exists(),
            "docker-compose.yml should exist in project root"
        )

    def test_dockerfile_exists(self):
        """Verify Dockerfile exists."""
        self.assertTrue(
            self.dockerfile.exists(),
            "Dockerfile should exist in project root"
        )

    def test_docker_compose_yaml_syntax(self):
        """Verify docker-compose.yml has valid YAML syntax."""
        try:
            import yaml
            with open(self.compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            self.assertIsNotNone(compose_config, "YAML should parse successfully")
        except ImportError:
            self.skipTest("PyYAML not installed, skipping YAML syntax test")
        except yaml.YAMLError as e:
            self.fail(f"docker-compose.yml has invalid YAML syntax: {e}")

    def test_service_name_is_swasthyasetu(self):
        """Verify service is named 'swasthyasetu'."""
        try:
            import yaml
            with open(self.compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            services = compose_config.get('services', {})
            self.assertIn(
                'swasthyasetu',
                services,
                "Service should be named 'swasthyasetu'"
            )
        except ImportError:
            self.skipTest("PyYAML not installed")

    def test_port_mapping_8000(self):
        """Verify port 8000 is mapped correctly."""
        try:
            import yaml
            with open(self.compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            service = compose_config['services']['swasthyasetu']
            ports = service.get('ports', [])
            
            self.assertTrue(
                any('8000:8000' in str(p) for p in ports),
                "Port 8000:8000 should be mapped"
            )
        except ImportError:
            self.skipTest("PyYAML not installed")

    def test_env_file_configuration(self):
        """Verify .env file is configured."""
        try:
            import yaml
            with open(self.compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            service = compose_config['services']['swasthyasetu']
            env_file = service.get('env_file', [])
            
            self.assertIn('.env', env_file, 
                "Should load environment variables from .env file")
        except ImportError:
            self.skipTest("PyYAML not installed")

    def test_required_volumes_mounted(self):
        """Verify required volumes are mounted."""
        try:
            import yaml
            with open(self.compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            service = compose_config['services']['swasthyasetu']
            volumes = service.get('volumes', [])
            volume_strs = [str(v) for v in volumes]
            
            required_volumes = [
                './static',
                './templates',
                './pubmed_data'
            ]
            
            for required in required_volumes:
                self.assertTrue(
                    any(required in v for v in volume_strs),
                    f"Volume '{required}' should be mounted"
                )
        except ImportError:
            self.skipTest("PyYAML not installed")

    def test_healthcheck_configured(self):
        """Verify healthcheck is configured for /health endpoint."""
        try:
            import yaml
            with open(self.compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            service = compose_config['services']['swasthyasetu']
            healthcheck = service.get('healthcheck', {})
            
            self.assertIn('test', healthcheck, 
                "Healthcheck test command should be defined")
            test_cmd = ' '.join(healthcheck.get('test', []))
            self.assertIn('/health', test_cmd,
                "Healthcheck should target /health endpoint")
        except ImportError:
            self.skipTest("PyYAML not installed")

    def test_restart_policy(self):
        """Verify restart policy is set to unless-stopped."""
        try:
            import yaml
            with open(self.compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            service = compose_config['services']['swasthyasetu']
            restart = service.get('restart', '')
            
            self.assertEqual(
                restart,
                'unless-stopped',
                "Restart policy should be 'unless-stopped'"
            )
        except ImportError:
            self.skipTest("PyYAML not installed")

    def test_persistent_volume_defined(self):
        """Verify persistent volume is defined."""
        try:
            import yaml
            with open(self.compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            volumes = compose_config.get('volumes', {})
            self.assertIn(
                'swasthyasetu-data',
                volumes,
                "Persistent volume 'swasthyasetu-data' should be defined"
            )
        except ImportError:
            self.skipTest("PyYAML not installed")

    def test_dockerfile_multi_stage_build(self):
        """Verify Dockerfile uses multi-stage build."""
        with open(self.dockerfile, 'r') as f:
            content = f.read()
        
        # Check for multi-stage indicators
        self.assertIn('AS builder', content,
            "Dockerfile should have a builder stage")
        self.assertIn('AS production', content,
            "Dockerfile should have a production stage")
        self.assertIn('--from=builder', content,
            "Dockerfile should copy from builder stage")

    def test_dockerfile_exposes_port_8000(self):
        """Verify Dockerfile exposes port 8000."""
        with open(self.dockerfile, 'r') as f:
            content = f.read()
        
        self.assertIn('EXPOSE 8000', content,
            "Dockerfile should expose port 8000")

    def test_dockerfile_healthcheck(self):
        """Verify Dockerfile has healthcheck."""
        with open(self.dockerfile, 'r') as f:
            content = f.read()
        
        self.assertIn('HEALTHCHECK', content,
            "Dockerfile should define a HEALTHCHECK")


class TestDockerSetupInstructions(unittest.TestCase):
    """Test that Docker setup documentation is complete."""

    def test_readme_contains_docker_section(self):
        """Verify README.md contains Docker instructions."""
        readme_path = Path(__file__).parent / "DOCKER_README.md"
        if not readme_path.exists():
            self.skipTest("README.md not found")
        
        with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        docker_keywords = ['docker', 'Docker', 'docker-compose', 'container']
        self.assertTrue(
            any(keyword in content for keyword in docker_keywords),
            "README should contain Docker-related instructions"
        )


def run_validation():
    """Run all validation tests."""
    # Check if docker-compose is available
    try:
        result = subprocess.run(
            ['docker-compose', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"[OK] Docker Compose found: {result.stdout.strip()}")
        else:
            print("[WARN] Docker Compose check returned non-zero exit code")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("[WARN] Docker Compose not found - skipping runtime validation")
    
    # Run unit tests
    print("\n" + "="*60)
    print("Running Docker Compose Configuration Tests")
    print("="*60 + "\n")
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDockerCompose)
    suite.addTests(loader.loadTestsFromTestCase(TestDockerSetupInstructions))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation()
    exit(0 if success else 1)
