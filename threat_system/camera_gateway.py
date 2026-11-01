"""Validate and probe the existing private-LAN Raspberry Pi camera gateway."""

import ipaddress
import json
import re
import socket
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def gateway_origin(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("Enter the Pi IP address, hostname, or camera gateway URL.")
    value = value.strip()
    parsed = urlsplit(value if "://" in value else "http://" + value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/", "/stream.mjpg") or parsed.query or parsed.fragment
            or not re.fullmatch(r"[a-zA-Z0-9.:-]+", parsed.hostname)):
        raise ValueError("Use a Pi address or http://<pi-ip>:8000/stream.mjpg, without credentials or query parameters.")
    try:
        port = parsed.port if parsed.port is not None else 8000
    except ValueError as error:
        raise ValueError("Gateway port must be between 1 and 65535.") from error
    if not 1 <= port <= 65535:
        raise ValueError("Gateway port must be between 1 and 65535.")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"{parsed.scheme}://{host}:{port}"


def resolve_gateway(origin):
    parsed = urlsplit(origin)
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("Could not resolve the Pi hostname. Try its LAN IP address.")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ((ip.is_private or ip.is_loopback) and not
                (ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved)):
            raise ValueError("Use a private LAN address for the Raspberry Pi gateway.")
    # Pin the checked IP so later requests cannot be redirected by a DNS change.
    ip = ipaddress.ip_address(addresses[0][4][0])
    host = f"[{ip}]" if ip.version == 6 else str(ip)
    return f"{parsed.scheme}://{host}:{parsed.port}"


def read_gateway_frame(origin):
    opener = build_opener(ProxyHandler({}), NoRedirects())
    deadline = time.monotonic() + 5
    with opener.open(origin + "/stream.mjpg", timeout=5) as response:
        if response.headers.get_content_type() != "multipart/x-mixed-replace":
            raise ValueError("The gateway did not return an MJPEG camera stream.")
        data = bytearray()
        while len(data) < 2 * 1024 * 1024 and time.monotonic() < deadline:
            chunk = response.read1(4096)
            if not chunk:
                break
            data.extend(chunk)
            start = data.find(b"\xff\xd8")
            end = data.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if end >= 0:
                return bytes(data[start:end + 2])
    raise ValueError("Gateway reachable, but no complete camera frame arrived. Check the Pi camera service.")


def check_gateway(origin):
    origin = resolve_gateway(origin)
    opener = build_opener(ProxyHandler({}), NoRedirects())
    with opener.open(origin + "/health", timeout=5) as response:
        data = response.read(4097)
        if len(data) > 4096:
            raise ValueError("Unexpected gateway health response.")
        health = json.loads(data)
    if not isinstance(health, dict) or health.get("service") != "threat-edge-camera" or health.get("status") != "ok":
        raise ValueError("This address is not the project's threat-edge-camera gateway.")
    return origin, read_gateway_frame(origin)
