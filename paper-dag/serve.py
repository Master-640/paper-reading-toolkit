# -*- coding: utf-8 -*-
"""
论文 DAG 阅读地图 —— 本地服务器启动器

用法：
    python serve.py            # 默认 8765 端口
    python serve.py 9000       # 指定端口

它会以「刘睿涵\\刘睿涵」为根目录起一个静态服务器，
所以 index.html 里 ../../../papers/*.pdf 这类相对链接全都能正常打开。
"""
import functools
import http.server
import os
import socketserver
import sys
import threading
import urllib.parse
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

rel = os.path.relpath(os.path.join(HERE, "index.html"), ROOT).replace("\\", "/")
url = "http://127.0.0.1:%d/%s" % (PORT, urllib.parse.quote(rel))

Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ROOT)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    with Server(("127.0.0.1", PORT), Handler) as httpd:
        print("=" * 62)
        print("  论文 DAG 阅读地图")
        print("=" * 62)
        print("  根目录 : %s" % ROOT)
        print("  打开   : %s" % url)
        print("  停止   : 按 Ctrl+C")
        print("=" * 62)
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止。")


if __name__ == "__main__":
    main()
