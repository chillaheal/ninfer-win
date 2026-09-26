import json, struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3_8_flash_next.ninfer"
needles = sys.argv[2:] or ["token_embedding"]

with open(path, "rb") as f:
    magic = f.read(8)
    json_bytes = struct.unpack("<Q", f.read(8))[0]
    data = f.read(json_bytes)
s = data.decode("utf-8", "replace")
print("magic:", magic, "json_bytes:", json_bytes, "read:", len(data))

j = json.loads(s)
objs = j.get("objects", [])
print("identity:", j.get("identity"))
print("object count:", len(objs))

for needle in needles:
    hits = [o for o in objs if needle in o.get("name", "")]
    print(f"\n### {needle}: {len(hits)} object(s)")
    for o in hits[:5]:
        print(json.dumps(o))
