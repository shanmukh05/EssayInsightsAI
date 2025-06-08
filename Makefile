FRONTEND_DIR = chatbot/frontend/app.py
BACKEND_FILE = run

.PHONY:  run_app setup run_frontend run_backend clean

setup:
	@echo "Setting up the environment..."
	@pip install -r requirements.txt
	@echo "Setup complete."

run_frontend:
	@echo "Starting frontend..."
	@streamlit run $(FRONTEND_DIR)

run_backend:
	@echo "Starting backend..."
	@cd chatbot/backend && uvicorn $(BACKEND_FILE):app --reload 

run_app: run_frontend run_backend

clean:
	@echo "Cleaning up..."
	@python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	@echo "Cleanup complete."