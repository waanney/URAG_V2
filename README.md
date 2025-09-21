# Tạo và kích hoạt môi trường + cài gói trong 1 dòng
uv venv
uv pip install -r requirements.txt

-> Chạy venv: source .venv/bin/activate

# Hoặc chạy trực tiếp mà không cần venv thủ công
uv run python script.pyu

export PYTHONPATH="$PWD"
