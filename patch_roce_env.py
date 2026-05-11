#!/usr/bin/env python3
from pathlib import Path
import re
import shutil
import time

ROOT = Path(".")
FILES = [
    ROOT / "examples" / "roce_coordinator.cpp",
    ROOT / "examples" / "legacy_roce_coordinator.cpp",
]

HELPER = r'''
static std::string getenv_str(const char* name, const std::string& def = "") {
  const char* v = std::getenv(name);
  return (v && *v) ? std::string(v) : def;
}

static int getenv_int(const char* name, int def = -1) {
  const char* v = std::getenv(name);
  if (!v || !*v) return def;
  return std::stoi(v);
}
'''

for path in FILES:
    if not path.exists():
        print(f"[skip] {path} not found")
        continue

    text = path.read_text()

    backup = path.with_suffix(path.suffix + f".bak_{int(time.time())}")
    shutil.copy2(path, backup)
    print(f"[backup] {path} -> {backup}")

    if "static std::string getenv_str" not in text:
        # Insert helper after using namespace std; if exists
        text = re.sub(
            r"(using namespace std;\s*)",
            r"\1\n" + HELPER + "\n",
            text,
            count=1,
        )

    # Replace hard-coded remote worker device/gid.
    text = text.replace(
        'Endpoint::roce(worker1_host, worker1_port, "rocep131s0f0", -1)',
        'Endpoint::roce(worker1_host, worker1_port, getenv_str("ROCE_WORKER1_DEVICE", cfg.device_name), getenv_int("ROCE_WORKER1_GID_INDEX", cfg.gid_index))'
    )

    # Optional: replace any remaining gid auto-select in worker endpoint with env override
    text = text.replace(
        'Endpoint::roce(local_worker_host, local_worker_port, cfg.device_name, cfg.gid_index)',
        'Endpoint::roce(local_worker_host, local_worker_port, getenv_str("ROCE_LOCAL_WORKER_DEVICE", cfg.device_name), getenv_int("ROCE_LOCAL_WORKER_GID_INDEX", cfg.gid_index))'
    )

    path.write_text(text)
    print(f"[patched] {path}")

print("[done] Rebuild project after patch.")