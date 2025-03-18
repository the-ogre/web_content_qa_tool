import unittest
from utils import validate_url, clean_text
from scraper import scrape_url

class TestWebContentQA(unittest.TestCase):
    
    def test_url_validation(self):
        """Test URL validation function"""
        # Valid URLs
        self.assertTrue(validate_url("https://www.example.com"))
        self.assertTrue(validate_url("http://example.com/path?query=value"))
        self.assertTrue(validate_url("https://subdomain.example.co.uk/path"))
        
        # Invalid URLs
        self.assertFalse(validate_url("not-a-url"))
        self.assertFalse(validate_url("www.example.com"))  # Missing protocol
        self.assertFalse(validate_url("http://"))  # Missing domain
    
    def test_text_cleaning(self):
        """Test text cleaning function"""
        # Test removing extra whitespace
        self.assertEqual(clean_text("  Hello   world  "), "Hello world")
        
        # Test with newlines and tabs
        self.assertEqual(clean_text("Hello\nworld\t!"), "Hello world !")
        
        # Test with empty input
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
    
    def test_scraper_with_valid_url(self):
        """Test scraper with a valid URL"""
        # This test will actually make a network request
        content = scrape_url("https://example.com")
        self.assertIsNotNone(content)
        self.assertIn("Example Domain", content)
    
    def test_scraper_with_invalid_url(self):
        """Test scraper with an invalid URL"""
        content = scrape_url("not-a-url")
        self.assertIsNone(content)

if __name__ == "__main__":
    unittest.main()