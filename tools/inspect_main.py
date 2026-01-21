import importlib,sys,traceback
try:
    m=importlib.import_module('main')
    keys=sorted(k for k in m.__dict__.keys() if not k.startswith('__'))
    print('total_keys', len(keys))
    print('HAS_app_in_dir', 'app' in keys)
    print('hasattr_app', hasattr(m,'app'))
    print('first_100_keys', keys[:100])
    if 'app' in keys:
        print('app_type', type(m.__dict__['app']))
except Exception:
    traceback.print_exc()
    sys.exit(2)
