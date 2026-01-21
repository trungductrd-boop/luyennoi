import os, sys
print('CWD:', os.path.abspath('.'))
print('LISTDIR:')
for p in os.listdir('.'):
    print(' ', p)
print('\nFIRST SYSPATH ENTRIES:')
for p in sys.path[:10]:
    print(' ', p)
# detect local numpy filename/dir
conflicts = [name for name in os.listdir('.') if name.lower().startswith('numpy')]
print('\nLOCAL NAMES STARTING WITH "numpy":', conflicts)
# try importing numpy and pydantic_core and print their __file__ if available
for mod in ('numpy','pydantic_core'):
    try:
        m = __import__(mod)
        print(f"{mod} imported OK, file=", getattr(m, '__file__', None))
    except Exception as e:
        print(f"{mod} import error: {e}")
print('\nPYTHON EXECUTABLE:', sys.executable)
print('PYTHON VERSION:', sys.version)
