import sys, traceback
sys.path.insert(0, r"d:\FINAL_PROJECT_v3\zewail_campus_assistant")
sys.path.insert(0, r"d:\FINAL_PROJECT_v3\zewail_campus_assistant\learning_analytics_xai\dashboard")
try:
    import analytics_page
    print("analytics_page: OK")
except Exception as e:
    print(f"analytics_page ERROR: {e}")
    traceback.print_exc()
