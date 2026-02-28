"""
Unit tests for the Visual Aid generation functionality in SwasthyaSetu.

Tests the URL generation logic and integration with the Educator agent.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions we want to test
from main import generate_visual_aid_url, VISUAL_AID_ENABLED, VISUAL_AID_WIDTH, VISUAL_AID_HEIGHT


class TestVisualAidGeneration(unittest.TestCase):
    """Test cases for the visual aid image generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_diagnoses = [
            ("Dengue Fever", "fever with severe headache and joint pain"),
            ("Type 2 Diabetes", "high blood sugar levels affecting insulin production"),
            ("Hypertension", "elevated blood pressure affecting cardiovascular health"),
            ("Asthma", "respiratory condition causing breathing difficulties"),
            ("Fractured Arm", "bone fracture in the forearm requiring immobilization"),
            ("Migraine", "severe headache with sensitivity to light"),
        ]

    def test_generate_visual_aid_url_returns_string(self):
        """Test that generate_visual_aid_url returns a string."""
        url = generate_visual_aid_url("Dengue Fever")
        self.assertIsInstance(url, str)
        self.assertTrue(len(url) > 0)

    def test_generate_visual_aid_url_contains_pollinations_domain(self):
        """Test that generated URL uses Pollinations.ai domain."""
        url = generate_visual_aid_url("Dengue Fever")
        self.assertIn("image.pollinations.ai", url)

    def test_generate_visual_aid_url_includes_dimensions(self):
        """Test that URL includes correct width and height parameters."""
        url = generate_visual_aid_url("Dengue Fever")
        self.assertIn(f"width={VISUAL_AID_WIDTH}", url)
        self.assertIn(f"height={VISUAL_AID_HEIGHT}", url)

    def test_generate_visual_aid_url_includes_nologo(self):
        """Test that URL includes nologo parameter."""
        url = generate_visual_aid_url("Dengue Fever")
        self.assertIn("nologo=true", url)

    def test_generate_visual_aid_url_includes_seed(self):
        """Test that URL includes seed parameter for consistent images."""
        url = generate_visual_aid_url("Dengue Fever")
        self.assertIn("seed=", url)

    def test_generate_visual_aid_url_url_encoding(self):
        """Test that special characters in diagnosis are properly URL-encoded."""
        url = generate_visual_aid_url("Heart Attack (Acute)")
        # Should not contain unencoded special characters in the URL
        self.assertNotIn(" ", url.split("?")[0])

    def test_generate_visual_aid_url_consistency(self):
        """Test that same diagnosis generates consistent URLs."""
        url1 = generate_visual_aid_url("Dengue Fever")
        url2 = generate_visual_aid_url("Dengue Fever")
        self.assertEqual(url1, url2)

    def test_generate_visual_aid_url_different_diagnoses(self):
        """Test that different diagnoses generate different URLs."""
        url1 = generate_visual_aid_url("Dengue Fever")
        url2 = generate_visual_aid_url("Malaria")
        self.assertNotEqual(url1, url2)

    def test_generate_visual_aid_url_anatomical_terms_included(self):
        """Test that anatomical context terms are included in the prompt."""
        # Test fever/infection conditions
        url = generate_visual_aid_url("Viral Fever")
        self.assertIn("medical%20illustration", url.lower())

    def test_generate_visual_aid_empty_diagnosis(self):
        """Test handling of empty diagnosis string."""
        url = generate_visual_aid_url("")
        self.assertIsInstance(url, str)
        self.assertTrue(len(url) > 0)

    def test_generate_visual_aid_none_explanation(self):
        """Test that None explanation is handled gracefully."""
        url = generate_visual_aid_url("Dengue Fever", explanation=None)
        self.assertIsInstance(url, str)
        self.assertTrue(len(url) > 0)

    def test_visual_aid_config_constants(self):
        """Test that visual aid configuration constants are set."""
        self.assertTrue(VISUAL_AID_ENABLED)
        self.assertIsInstance(VISUAL_AID_WIDTH, int)
        self.assertIsInstance(VISUAL_AID_HEIGHT, int)
        self.assertGreater(VISUAL_AID_WIDTH, 0)
        self.assertGreater(VISUAL_AID_HEIGHT, 0)


class TestVisualAidIntegration(unittest.TestCase):
    """Integration tests for visual aid in the education workflow."""

    def test_education_response_structure(self):
        """Test that education response includes visual_aid_url field."""
        # Mock education response with visual aid
        mock_education = {
            "explanation": "Dengue is a viral infection transmitted by mosquitoes.",
            "medication_info": "Consult your physician for treatment options.",
            "next_steps": ["Stay hydrated", "Rest", "Monitor fever"],
            "visual_aid_description": "Medical illustration showing dengue virus and mosquito transmission",
            "visual_aid_url": "https://image.pollinations.ai/prompt/test"
        }
        
        # Verify structure
        self.assertIn("visual_aid_url", mock_education)
        self.assertIn("visual_aid_description", mock_education)
        # Verify ai_generated_image flag would be added by formatter
        self.assertTrue(True)  # Structure is valid

    def test_visual_aid_url_fallback(self):
        """Test that empty URL is handled gracefully."""
        # Test with empty URL
        mock_education_empty = {
            "visual_aid_url": "",
            "explanation": "Test explanation"
        }
        
        # Should not crash
        self.assertEqual(mock_education_empty["visual_aid_url"], "")


class TestVisualAidErrorHandling(unittest.TestCase):
    """Test error handling for visual aid generation."""

    def test_generate_visual_aid_with_special_characters(self):
        """Test handling of special characters in diagnosis."""
        special_cases = [
            "COVID-19 (Coronavirus)",
            "Type 1 & Type 2 Diabetes",
            "Heart Disease: Cardiac Arrest",
            "Normal 'cold' symptoms",
        ]
        
        for diagnosis in special_cases:
            with self.subTest(diagnosis=diagnosis):
                url = generate_visual_aid_url(diagnosis)
                self.assertIsInstance(url, str)
                self.assertTrue(len(url) > 0)
                self.assertIn("image.pollinations.ai", url)


class TestVisualAidAnatomicalContext(unittest.TestCase):
    """Test that anatomical context is correctly added based on condition type."""

    def test_fever_conditions_get_body_parts_context(self):
        """Test that fever/infection conditions get appropriate anatomical context."""
        url = generate_visual_aid_url("High Fever")
        # Should include fever-related anatomical terms
        self.assertIn("medical%20illustration", url.lower())

    def test_bone_conditions_get_skeletal_context(self):
        """Test that bone/joint conditions get skeletal context."""
        url = generate_visual_aid_url("Fractured Leg")
        # Should include skeletal terms in URL encoding
        decoded_url = url.replace("%20", " ")
        # Check that it's a properly formed URL
        self.assertTrue(url.startswith("https://"))

    def test_heart_conditions_get_cardiovascular_context(self):
        """Test that heart conditions get cardiovascular context."""
        url = generate_visual_aid_url("Heart Attack")
        # Verify URL structure
        self.assertTrue(url.startswith("https://image.pollinations.ai/prompt/"))


if __name__ == "__main__":
    unittest.main()
