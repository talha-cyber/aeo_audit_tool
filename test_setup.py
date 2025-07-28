#!/usr/bin/env python3
"""
Quick test script to verify the FastAPI app and Celery worker setup
"""

def test_fastapi_import():
    """Test that FastAPI app can be imported"""
    try:
        from app.main import app
        print("✅ FastAPI app imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import FastAPI app: {e}")
        return False

def test_celery_import():
    """Test that Celery worker can be imported"""
    try:
        from celery_worker import celery_app, test_task
        print("✅ Celery worker imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Celery worker: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        if response.status_code == 200 and response.json() == {"status": "ok"}:
            print("✅ Health endpoint working correctly")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code} - {response.json()}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing AEO Audit Tool setup...\n")
    
    results = []
    results.append(test_fastapi_import())
    results.append(test_celery_import())
    results.append(test_health_endpoint())
    
    print(f"\nResults: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Create a .env file with your configuration")
        print("2. Run: docker-compose up -d")
        print("3. Visit: http://localhost:8000/health")
        print("4. View API docs: http://localhost:8000/docs")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.") 