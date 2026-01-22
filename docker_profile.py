import tracemalloc
import psutil
import time
import importlib

tracemalloc.start()
proc = psutil.Process()

rss_before = proc.memory_info().rss
print('RSS before importing project modules:', rss_before)

try:
    import main
    importlib.reload(main)
    time.sleep(0.5)
except Exception as e:
    print('Import error:', repr(e))

rss_after = proc.memory_info().rss
print('RSS after importing project modules:', rss_after)
print('Delta RSS (bytes):', rss_after - rss_before)

snapshot = tracemalloc.take_snapshot()
stats = snapshot.statistics('lineno')[:15]
print('\nTop tracemalloc allocations:')
for stat in stats:
    print(stat)

tracemalloc.stop()

# Keep container alive briefly for inspection (if running interactively)
time.sleep(1)