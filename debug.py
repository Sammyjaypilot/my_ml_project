import sys
import os
sys.path.append('.')

# Try to import the entire module first
try:
    import src.ml.model_loader as model_loader
    print("✅ Module imported successfully")
    
    # Check what functions are available
    print("Available functions:", [f for f in dir(model_loader) if not f.startswith('_')])
    
    # Try to call get_model_info directly
    if hasattr(model_loader, 'get_model_info'):
        print("✅ get_model_info function exists")
        print(model_loader.get_model_info())
    else:
        print("❌ get_model_info function NOT found")
        
except Exception as e:
    print(f"❌ Module import failed: {e}")
    import traceback
    traceback.print_exc()
