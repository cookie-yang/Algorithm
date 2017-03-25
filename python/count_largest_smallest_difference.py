n = input()
n = input()
n = n.split()
print(n)
m= [int(n[i]) for i in range(len(n))]
m.sort()
print(m)
minV = 99999
maxV = m[-1]-m[0]
minPair = 0
maxPair1 = 0
maxPair2 = 0
for x in range(1,len(m)):
    print(m[x])
    if m[x]==m[0]:
        maxPair1+=1
    if m[x]==m[-1]:
        maxPair2+=1
    if m[x]-m[x-1]==minV:
        minPair +=1
    elif m[x]-m[x-1]<minV:
        minV = m[x]-m[x-1]
        minPair = 1
print(minPair," ",max(maxPair1,maxPair2))
