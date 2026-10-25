import sys
p = sys.argv[1]; s = open(p).read()
a = '  echo "$q" | grep -qx wb-k12-L1 || retired L1 || {'
i = s.index(a); j = s.index('\n', i)
line = s[i:j]
assert 'L1' in line
new = line.replace('wb-k12-L1 || retired L1', 'wb-k12-L2 || retired L2').replace('-J wb-k12-L1', '-J wb-k12-L2').replace('resubmitted L1', 'resubmitted L2')
s = s[:j] + '\n' + new + s[j:]
open(p, "w").write(s); print("patched")
