FRONTEND_DIR = chatbot/frontend/app.py

.PHONY: run_app setup clean

setup:
	@echo "Setting up the environment..."
	@pip install -r requirements.txt
	@echo "Setup complete."

run_app:
	@echo "Starting frontend..."
	@streamlit run $(FRONTEND_DIR)


clean:
	@echo "Cleaning up..."
	@python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	@echo "Cleanup complete."