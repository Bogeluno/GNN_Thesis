from tqdm import tqdm
s = 0

for i in tqdm(range(50000000)):
    s += (-1)**i/(2*i+1)

print(s*4)