# launches interpviz at http://localhost:8000, restarting the server whenever a .py file under interpviz/ changes
# (the loaded model is dropped on restart: click load again; frontend changes only need a browser refresh)
# run from repo root: venv/bin/python -m interpviz.app

import uvicorn

PORT = 8000

if __name__ == "__main__":
    # reload needs the app as an import string, so the worker process can re-import it
    uvicorn.run("interpviz.back.server:app", host="127.0.0.1", port=PORT, reload=True, reload_dirs=["interpviz"])
