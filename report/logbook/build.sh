#!/bin/bash
# usage: build.sh  (reads p32.txt, p32fig.txt, r4end.txt, r4dur.txt if present)
cd "$(dirname "$0")"
python3 - <<'P'
import os,datetime
s=open('logbook.src.html').read()
def rd(f,d): return open(f).read().strip() if os.path.exists(f) else d
today=datetime.date.today().strftime('%-d %B %Y')
s=s.replace('{{P32}}',rd('p32.txt','')).replace('{{P32FIG}}',rd('p32fig.txt',''))
s=s.replace('{{R4END}}',rd('r4end.txt','in progress')).replace('{{R4DUR}}',rd('r4dur.txt',''))
s=s.replace('{{ENDDATE}}',rd('enddate.txt',today)).replace('{{COMPILED}}',today)
open('logbook.html','w').write(s)
P
chromium --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer --virtual-time-budget=15000 --run-all-compositor-stages-before-draw --print-to-pdf=NAS_WHITEBOX_Logbook.pdf "file://$PWD/logbook.html" 2>/dev/null
pdfinfo NAS_WHITEBOX_Logbook.pdf | grep -E "Pages|File size"; pdffonts NAS_WHITEBOX_Logbook.pdf | awk 'NR>2{print $1}' | sort -u | tr '\n' ' '
