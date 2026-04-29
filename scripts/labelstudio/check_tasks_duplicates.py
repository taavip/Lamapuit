#!/usr/bin/env python3
import json, subprocess, sys

refresh='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA4MTkxNjE4OSwiaWF0IjoxNzc0NzE2MTg5LCJqdGkiOiI2YjgxNzBhOTgyYTc0YjhmOTIwOTIxZWEwNmIwMjQ1MCIsInVzZXJfaWQiOiIxIn0.SNvdYAM2i27B9bDV_y557tk-GM8SReude1P7xr5zpvw'

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print('ERROR', cmd, p.stderr)
        sys.exit(2)
    return p.stdout

def main():
    tok_json = run(['curl','-s','-X','POST','-H','Content-Type: application/json','-d', json.dumps({'refresh':refresh}), 'http://localhost:8080/api/token/refresh'])
    access = json.loads(tok_json).get('access')
    tasks_json = run(['curl','-s','-H', f'Authorization: Bearer {access}', 'http://localhost:8080/api/tasks?project=7&limit=10000'])
    obj = json.loads(tasks_json)
    tasks = obj.get('results', obj if isinstance(obj, list) else [])
    print('TASKS', len(tasks))
    m = {}
    for t in tasks:
        img = t.get('data',{}).get('image')
        m.setdefault(img, []).append(t.get('id'))
    dups = {k:v for k,v in m.items() if len(v)>1}
    print('DUPLICATE_IMAGES', len(dups))
    if dups:
        for k,v in list(dups.items())[:20]:
            print(k, v)

if __name__ == '__main__':
    main()
