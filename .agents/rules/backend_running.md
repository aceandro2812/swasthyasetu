# Running the Backend

Always run the backend through uvicorn and ensure the `uv` virtual environment is activated before running. We use `uv` for package management and virtual environments in this project.

## Workflow

1. **Activate the environment**: 
   - Windows: `.\.venv\Scripts\activate`
   - Unix/macOS: `source .venv/bin/activate`

2. **Installing packages**:
   - MUST use `uv pip install <package_name>` instead of standard pip. This ensures faster resolution and compatibility.

3. **Running the server**:
   `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
