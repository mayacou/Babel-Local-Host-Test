#!/bin/bash
echo "Running translation tests..."

# Run tests
# pytest -v -s tests/test_europarl_small100.py
# rm -rf ~/.cache/huggingface/hub  # Delete Hugging Face cache

# pytest -v -s tests/test_tedTalk_small100.py
# rm -rf ~/.cache/huggingface/hub  # Delete Hugging Face cache

# pytest -v -s tests/test_wmt_small100.py
rm -rf ~/.cache/huggingface/hub  # Delete Hugging Face cache

python3 -m tests.test_Helsinki.py
rm -rf ~/.cache/huggingface/hub  # Delete Hugging Face cache

pytest -v -s tests/test_M2M.py
rm -rf ~/.cache/huggingface/hub  # Delete Hugging Face cache

pytest -v -s tests/test_NLLB.py
rm -rf ~/.cache/huggingface/hub  # Delete Hugging Face cache

python3 -m tests.test_towerinstruct.py
rm -rf ~/.cache/huggingface/hub  # Delete Hugging Face cache

python3 -m tests.test_googletranslate.py
rm -rf ~/.cache/huggingface/hub  # Delete Hugging Face cache

python3 -m tests.test_gemini.py
rm -rf ~/.cache/huggingface/hub  # Delete Hugging Face cache

python3 -m tests.test_chatGPT.py
rm -rf ~/.cache/huggingface/hub  # Delete Hugging Face cache
