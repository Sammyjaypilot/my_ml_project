import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_health_check():
    """Basic test to verify CI/CD is working"""
    assert 1 + 1 == 2

def test_imports():
    """Test that main modules can be imported"""
    try:
        from api.main import app
        assert True
    except ImportError:
        assert False, "Failed to import main app"
