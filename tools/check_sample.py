import json
import helpers

if __name__ == '__main__':
    sid = '0b5f1e23'
    meta = helpers.get_sample_metadata(sid)
    print(json.dumps({"sample_id": sid, "meta": meta}, ensure_ascii=False, indent=2))
