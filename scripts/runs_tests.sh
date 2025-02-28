# Make it executable:
# chmod +x scripts/run_tests.sh
# run it:
# ./scripts/run_tests.sh
#!/bin/bash
echo "Running translation tests..."
pytest tests/test_wmt.py