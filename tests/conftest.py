import pathlib, socket, sys, threading, time

import httpx, pytest, uvicorn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from envs.invoice_env import build_invoice_env  # noqa: E402
from mockco.app import app  # noqa: E402


@pytest.fixture(scope="session")
def base_url():
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try: httpx.get(f"http://127.0.0.1:{port}/"); break
        except httpx.HTTPError: time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


@pytest.fixture(scope="session")
def env(base_url, tmp_path_factory):
    reg, session = build_invoice_env(base_url, str(ROOT / "mockco/data/invoices"), str(tmp_path_factory.mktemp("art")))
    yield reg, session
    session.close()


@pytest.fixture(autouse=True)
def fresh(base_url):
    httpx.post(f"{base_url}/__admin/reset")
