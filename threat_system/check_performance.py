"""Quick performance check after improvements"""
import json
import subprocess
import os
import sys

videos = ['weapon.mp4', 'f1.mp4', 'f2.mp4', 'loitering.mp4']

print('Performance after visualization & consistency improvements:')
print('='*55)

for video in videos:
    if os.path.exists('results/output.json'):
        os.remove('results/output.json')
    
    try:
        subprocess.run(
            [sys.executable, 'main.py', '--video', video, '--output', 'results'],
            check=True, capture_output=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f'{video:20} FAIL: {exc}')
        continue
    
    with open('results/output.json') as f:
        data = json.load(f)
    
    total = 0
    weapons = 0
    for frame in data:
        for person in frame.get('persons', []):
            total += 1
            if person.get('weapon_present'):
                weapons += 1
    
    pct = 100*weapons/total if total > 0 else 0
    status = 'GOOD' if ('weapon' in video and pct > 3) or ('weapon' not in video and pct < 5) else 'CHECK'
    print(f'{video:20} {weapons:3}/{total:4} ({pct:5.1f}%) - {status}')

print('='*55)
