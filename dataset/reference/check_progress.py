import os

log_path = "/liyang/works/nowcast_case/experiments/DiffCast_full/run.log"

with open(log_path, 'rb') as f:
    content = f.read()

print(f"Total bytes: {len(content)}")

last_4k = content[-4096:] if len(content) > 4096 else content
text = last_4k.decode('utf-8', errors='replace')

lines = text.split('\n')
for i, line in enumerate(lines[-30:]):
    cleaned = line.replace('\r', '\n').strip()
    if cleaned:
        for part in cleaned.split('\n'):
            part = part.strip()
            if part:
                print(f"  {part}")

# Also check for any \r delimited progress bars in the last chunk
last_chunk = content[-2048:].decode('utf-8', errors='replace')
parts = last_chunk.split('\r')
if len(parts) > 1:
    print(f"\nFound {len(parts)} \\r-separated segments. Last 3:")
    for p in parts[-3:]:
        p = p.strip()
        if p:
            print(f"  [{p[:200]}]")

# Check tensorboard events
import struct
events_dir = "/liyang/works/nowcast_case/experiments/DiffCast_full/lightning_logs/radar_nowcast/version_0"
for f in os.listdir(events_dir):
    fp = os.path.join(events_dir, f)
    size = os.path.getsize(fp)
    print(f"\n{f}: {size} bytes")
