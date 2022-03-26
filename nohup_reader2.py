# 保留测试结果

filename = '20220325_005710'

with open(filename + '.log', 'r') as r:
    lines = r.readlines()
with open(filename + '-r2.txt', 'w') as w:
    for line in lines:
        if line[26:52] == 'mmdet - INFO - Epoch(val) ':
            w.write(line)
            print(line)

