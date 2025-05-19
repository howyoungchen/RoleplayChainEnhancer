#!/usr/bin/env python3
import json
import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

class DebugHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # 收集请求数据
        request_data = {
            "path": self.path,
            "headers": dict(self.headers),
        }
        length = int(self.headers.get('Content-Length', 0))
        body_bytes = self.rfile.read(length)
        try:
            body_str = body_bytes.decode('utf-8')
            try:
                request_data["body"] = json.loads(body_str) # 尝试解析为JSON
            except json.JSONDecodeError:
                request_data["body"] = body_str # 如果不是JSON，则按原样保存
        except UnicodeDecodeError:
            request_data["body"] = body_bytes.hex() # 如果无法解码为UTF-8，则保存为十六进制字符串
            request_data["body_encoding_error"] = "Failed to decode body as UTF-8, saved as hex."

        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"request_{timestamp}.json"

        # 将数据写入JSON文件
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(request_data, f, ensure_ascii=False, indent=4)
            print(f"Request data saved to {filename}")
        except Exception as e:
            print(f"Error saving request data to {filename}: {e}")

        # 返还一个简单响应
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(b"OK")

if __name__ == '__main__':
    addr = ('127.0.0.1', 8788)
    print(f"Debug server running on http://{addr[0]}:{addr[1]}")
    HTTPServer(addr, DebugHandler).serve_forever()
