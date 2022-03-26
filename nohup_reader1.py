# 删除进度条

filename = 'nohup'

with open(filename + '.log', 'r') as r:
    lines = r.readlines()
with open(filename + '-r1.txt', 'w') as w:
    for line in lines:
        if line[0] != '[':
            w.write(line)

