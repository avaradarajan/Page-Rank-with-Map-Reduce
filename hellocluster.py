import sys
import os
print("Hello CLuster")
file = sys.argv[1]
with open(file) as fp:
    for line in fp:
        print(line)
        print("--")
