import os, datetime

pid = 1032269
try:
    os.kill(pid, 0)
    print(f"Process {pid}: RUNNING")
except ProcessLookupError:
    print(f"Process {pid}: NOT RUNNING")

print(f"Current time: {datetime.datetime.now()}")

log = "/liyang/works/nowcast_case/experiments/DiffCast_full/run.log"
size = os.path.getsize(log)
mtime = datetime.datetime.fromtimestamp(os.path.getmtime(log))
print(f"Log: {size} bytes, modified: {mtime}")

with open(log, 'rb') as f:
    content = f.read()
text = content.decode('utf-8', errors='replace')

for line in text.split('\n'):
    for part in line.split('\r'):
        p = part.strip()
        if p and ('val metrics' in p or 'test metrics' in p or 'ERROR' in p or 'WARNING' in p or 'NaN' in p):
            print(p)
