with open('nohup.log', 'r') as r:
    lines = r.readlines()
with open('nohup2.txt', 'w') as w:
    for line in lines:
        if line[0] != '[':
            w.write(line)
