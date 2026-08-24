import os

log_path = "/liyang/works/nowcast_case/experiments/DiffCast_full/run.log"
with open(log_path, 'rb') as f:
    content = f.read()

print(f"Total bytes: {len(content)}")
text = content.decode('utf-8', errors='replace')

# Split by \r and \n to find all distinct lines
all_parts = []
for line in text.split('\n'):
    for part in line.split('\r'):
        part = part.strip()
        if part:
            all_parts.append(part)

# Print unique info lines (skip progress bars)
seen = set()
for p in all_parts:
    if 'INFO' in p or 'ERROR' in p or 'WARNING' in p or 'Traceback' in p or 'Exception' in p or 'early' in p.lower() or 'stop' in p.lower() or 'Epoch' in p:
        if p not in seen:
            seen.add(p)
            print(p)
