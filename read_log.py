import sys

try:
    with open('output.log', 'r', encoding='utf-16LE') as f:
        content = f.read()
except UnicodeError:
    with open('output.log', 'r', encoding='utf-8') as f:
        content = f.read()
        
lines = content.splitlines()[-50:]
for line in lines:
    print(line)
