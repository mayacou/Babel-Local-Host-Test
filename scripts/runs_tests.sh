# Make it executable:
# chmod +x scripts/run_tests.sh
# run it:
# ./scripts/run_tests.sh
#!/bin/bash
echo "Running translation tests..."
pytest -v -s tests/test_europarl_small100.py
huggingface-cli delete-cache
pytest -v -s tests/test_tedTalk_small100.py
huggingface-cli delete-cache
pytest -v -s tests/test_wmt_small100.py
huggingface-cli delete-cache
python3 -m tests.test_Helsinki.py
huggingface-cli delete-cache
pytest -v -s tests/test_M2M.py
huggingface-cli delete-cache
pytest -v -s tests/test_NLLB.py
huggingface-cli delete-cache
python3 -m tests.test_towerinstruct.py
huggingface-cli delete-cache
python3 -m tests.test_googletranslate.py
huggingface-cli delete-cache
python3 -m tests.test_gemini.py
huggingface-cli delete-cache
python3 -m tests.test_chatGPT.py
huggingface-cli delete-cache

