import re
p = r"C:\Program Files (x86)\Zwift\ZwiftApp.exe"
b = open(p, "rb").read()
t = b.decode("latin-1")
seen = set()
for m in re.finditer(r"(?i)levelup|frameupgrade|upgradecategory|_highend|_midrange|_entry|distance_|elevation_|time_", t):
    s = max(0, m.start()-40); e = min(len(t), m.end()+40)
    frag = re.sub(r"[^\x20-\x7e]", ".", t[s:e])
    key = frag.strip(".")
    if key and key not in seen:
        seen.add(key)
        print(frag)
    if len(seen) > 80:
        break
