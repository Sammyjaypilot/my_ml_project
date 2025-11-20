import sys
import os
sys.path.append('.')

try:
    from src.ml.model_loader import get_model_info
    print("✅ get_model_info imported successfully!")
    print(get_model_info())
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
