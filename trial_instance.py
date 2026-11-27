"""Control only the Streamlit instance that owns this trial directory."""
import argparse
from contextlib import contextmanager
import hashlib
import hmac
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import socket
import sys
import threading
import time
import webbrowser

from trial_build import BUILD_LABEL

HOST = "127.0.0.1"
STATE_FILE = ".trial-instance.json"


def state_for(root):
    try:
        state = json.loads((root / STATE_FILE).read_text(encoding="utf-8"))
        if (isinstance(state, dict) and state.get("root") == str(root)
                and isinstance(state.get("key"), str) and len(state["key"]) == 64
                and all(type(state.get(k)) is int and 0 < state[k] <= 65535
                        for k in ("control", "port"))):
            return state
    except (OSError, ValueError):
        pass
    return None


def tag(state, *values):
    message = "\n".join(values).encode()
    return hmac.new(state["key"].encode(), message, hashlib.sha256).hexdigest()


def status(state, action="status"):
    if not state:
        return None
    nonce = secrets.token_hex(16)
    conn = HTTPConnection(HOST, state["control"], timeout=2)
    try:
        conn.request("POST", "/" + action, headers={
            "X-Nonce": nonce, "X-Signature": tag(state, action, nonce)})
        reply = conn.getresponse()
        if reply.status == 200:
            value = reply.read(64).decode("ascii")
            if hmac.compare_digest(reply.getheader("X-Signature", ""), tag(state, action, nonce, value)):
                return value
    except (OSError, HTTPException, ValueError):
        pass
    finally:
        conn.close()
    return None


def port_in_use(port):
    with socket.socket() as probe:
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind((HOST, port))
        except OSError:
            return True
    return False


def own_app_ready(port):
    from streamlit import config
    from streamlit.runtime import Runtime
    from streamlit.runtime.runtime import RuntimeState
    # Streamlit 1.50 binds its socket before starting this process's runtime.
    if (not Runtime.exists() or Runtime.instance().state not in
            {RuntimeState.NO_SESSIONS_CONNECTED, RuntimeState.ONE_OR_MORE_SESSIONS_CONNECTED}
            or config.get_option("server.address") != HOST or config.get_option("server.port") != port):
        return False
    conn = HTTPConnection(HOST, port, timeout=1)
    try:
        conn.request("GET", "/_stcore/health")
        result = conn.getresponse()
        return result.status == 200 and result.read(16) == b"ok"
    except (OSError, HTTPException):
        return False
    finally:
        conn.close()


def make_control(state, stopping):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            action = self.path.lstrip("/")
            nonce = self.headers.get("X-Nonce", "")
            signature = self.headers.get("X-Signature", "")
            if (action not in {"status", "stop"} or len(nonce) != 32 or not nonce.isascii()
                    or not signature.isascii() or not hmac.compare_digest(signature, tag(state, action, nonce))):
                self.send_error(403)
                return
            if action == "stop":
                stopping.set()
            value = ("stopping" if stopping.is_set() else
                     "ready" if own_app_ready(state["port"]) else "starting")
            self.send_response(200)
            self.send_header("Content-Length", str(len(value)))
            self.send_header("X-Signature", tag(state, action, nonce, value))
            self.end_headers()
            self.wfile.write(value.encode("ascii"))

    return ThreadingHTTPServer((HOST, 0), Handler)


@contextmanager
def start_lock(root):
    """Windows releases this one-byte lock even when the console is closed."""
    import msvcrt
    with (root / ".trial-instance.lock").open("a+b") as lock:
        if not lock.tell():
            lock.write(b"1")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            yield False
        else:
            try:
                yield True
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def show_url(port):
    url = f"http://{HOST}:{port}"
    print(f"流哨已就绪。浏览器地址 {url}", flush=True)
    if not os.environ.get("LIUSHAO_NONINTERACTIVE"):
        webbrowser.open(url)


def monitor(state, stopping, finished):
    from streamlit.runtime import Runtime
    from streamlit.runtime.runtime import RuntimeState
    deadline = time.monotonic() + 90
    opened = False
    while not finished.wait(.25):
        if stopping.is_set():
            if Runtime.exists() and Runtime.instance().state != RuntimeState.INITIAL:
                Runtime.instance().stop()
                return
        elif not opened and own_app_ready(state["port"]):
            opened = True
            show_url(state["port"])
        elif not opened and time.monotonic() >= deadline:
            print("等待流哨就绪超时。请查看启动窗口的错误，关闭后运行环境自检。", flush=True)
            stopping.set()


def start(root, port):
    with start_lock(root) as acquired:
        if not acquired:
            deadline = time.monotonic() + 5
            while True:
                state = state_for(root)
                current = status(state)
                if current == "ready":
                    show_url(state["port"])
                    return 0
                if current or time.monotonic() >= deadline:
                    print("本目录的流哨正在启动或关闭。请查看原启动窗口，稍后再试。")
                    return 1
                time.sleep(.25)
        if port_in_use(port):
            print(f"端口 {port} 已由其他程序占用。请手动释放端口后再启动流哨。")
            return 1
        manifest = json.loads((root / "RELEASE_MANIFEST.json").read_bytes())
        anchors = manifest["artifacts"]
        os.environ["LIUSHAO_EXPECTED_MODEL_SHA256"] = anchors["model_sha256"]
        os.environ["LIUSHAO_EXPECTED_DESCRIPTOR_SHA256"] = anchors["model_descriptor_sha256"]
        state = {"root": str(root), "port": port, "key": secrets.token_hex(32), "pid": os.getpid()}
        stopping, finished = threading.Event(), threading.Event()
        with make_control(state, stopping) as control:
            state["control"] = control.server_port
            state_path = root / STATE_FILE
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state), encoding="utf-8")
            os.replace(temporary, state_path)
            try:
                threading.Thread(target=control.serve_forever, daemon=True).start()
                threading.Thread(target=monitor, args=(state, stopping, finished), daemon=True).start()
                from liushao_launch import main
                sys.argv = ["liushao", "--offline", "--artifacts", str(root / "artifacts"),
                            "--review-db", str(root / "trial-reviews.sqlite3"), "--port", str(port)]
                print(f"正在启动流哨 {BUILD_LABEL}。请保持此窗口打开。", flush=True)
                return main()
            finally:
                finished.set()
                control.shutdown()
                state_path.unlink(missing_ok=True)


def stop(root, port):
    state = state_for(root)
    if status(state, "stop") != "stopping":
        if port_in_use(port):
            print(f"端口 {port} 已占用，无法验证为本目录的流哨，未关闭任何程序。")
            return 1
        print("本目录没有正在运行的流哨。")
        return 0
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if state_for(root) != state:
            print("本目录的流哨已关闭。")
            return 0
        time.sleep(.2)
    print("已请求流哨退出，仍在等待它完成关闭。请查看原启动窗口。")
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    try:
        action = start if args.action == "start" else stop
        return action(args.root.resolve(), args.port)
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"无法{ '启动' if args.action == 'start' else '关闭' }流哨。请运行环境自检。详细信息：{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
