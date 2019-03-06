import csv

with open('task_to_vol.txt') as f:
    with open('task_vol.csv', 'w') as f2:
        writer = csv.writer(f2,delimiter=',')
        for line in f.readlines():
            task = int(line[:line.find(':')])
            vol = int(line[line.find(':')+1:])
            writer.writerow([task,vol])
