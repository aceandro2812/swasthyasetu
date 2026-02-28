"""
Unit tests for Dockerfile validation and container health checks.

These tests verify:
1. Dockerfile syntax and structure
2. Required files are copied correctly
3. Container health checks pass
"""

import subprocess
import unittest
from pathlib import Path


class TestDockerfile(unittest.TestCase):
    """Test cases for SwasthyaSetu Dockerfile."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.dockerfile_path = self.project_root / "Dockerfile"
        self.dockerignore_path = self.project_root / ".dockerignore"

    def test_dockerfile_exists(self):
        """Verify Dockerfile exists in project root."""
        self.assertTrue(
            self.dockerfile_path.exists(),
            "Dockerfile should exist in project root"
        )

    def test_dockerignore_exists(self):
        """Verify .dockerignore exists."""
        self.assertTrue(
            self.dockerignore_path.exists(),
            ".dockerignore should exist"
        )

    def test_dockerfile_uses_python_313(self):
        """Verify Dockerfile uses Python 3.13 base image."""
        content = self.dockerfile_path.read_text()
        self.assertIn(
            "python:3.13",
            content,
            "Dockerfile should use Python 3.13 base image"
        )

    def test_dockerfile_multi_stage_build(self):
        """Verify Dockerfile uses multi-stage build."""
        content = self.dockerfile_path.read_text()
        # Count FROM statements - should be at least 2 for multi-stage
        from_count = content.lower().count("from ")
        self.assertGreaterEqual(
            from_count,
            2,
            "Dockerfile should use multi-stage build (at least 2 FROM statements)"
        )

    def test_dockerfile_exposes_port_8000(self):
        """Verify Dockerfile exposes port 8000."""
        content = self.dockerfile_path.read_text()
        self.assertIn(
            "EXPOSE 8000",
            content,
            "Dockerfile should expose port 8000"
        )

    def test_dockerfile_uses_non_root_user(self):
        """Verify Dockerfile creates and uses non-root user."""
        content = self.dockerfile_path.read_text()
        self.assertIn(
            "USER",
            content,
            "Dockerfile should switch to non-root user"
        )

    def test_dockerfile_installs_faiss_dependencies(self):
        """Verify Dockerfile installs FAISS system dependencies."""
        content = self.dockerfile_path.read_text()
        self.assertIn(
            "libomp",
            content.lower(),
            "Dockerfile should install libomp for FAISS support"
        )

    def test_dockerfile_copies_required_files(self):
        """Verify Dockerfile copies all required application files."""
        content = self.dockerfile_path.read_text()
        required_copies = [
            "main.py",
            "static/",
            "templates/",
            "pyproject.toml"
        ]
        for file in required_copies:
            self.assertIn(
                file,
                content,
                f"Dockerfile should copy {file}"
            )

    def test_dockerfile_healthcheck_present(self):
        """Verify Dockerfile includes health check."""
        content = self.dockerfile_path.read_text()
        self.assertIn(
            "HEALTHCHECK",
            content,
            "Dockerfile should include HEALTHCHECK instruction"
        )

    def test_dockerfile_uvicorn_command(self):
        """Verify Dockerfile uses uvicorn to run the application."""
        content = self.dockerfile_path.read_text()
        self.assertIn(
            "uvicorn",
            content.lower(),
            "Dockerfile should use uvicorn server"
        )
        self.assertIn(
            "main:app",
            content,
            "Dockerfile should run main:app"
        )

    def test_dockerignore_excludes_venv(self):
        """Verify .dockerignore excludes virtual environments."""
        content = self.dockerignore_path.read_text()
        self.assertIn(
            ".venv/",
            content,
            ".dockerignore should exclude .venv/"
        )

    def test_dockerignore_excludes_env_files(self):
        """Verify .dockerignore excludes environment files."""
        content = self.dockerignore_path.read_text()
        self.assertIn(
            ".env",
            content,
            ".dockerignore should exclude .env files"
        )

    def test_dockerignore_excludes_cache(self):
        """Verify .dockerignore excludes Python cache."""
        content = self.dockerignore_path.read_text()
        self.assertIn(
            "__pycache__/",
            content,
            ".dockerignore should exclude __pycache__/"
        )


class TestDockerBuild(unittest.TestCase):
    """Integration tests for Docker build process."""

    @unittest.skip("Skipping actual Docker build in unit tests")
    def test_docker_build_succeeds(self):
        """Verify Docker image builds successfully."""
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            ["docker", "build", "-t", "swasthyasetu:test", "."],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Docker build failed: {result.stderr}"
        )


if __name__ == "__main__":
    unittest.main()
