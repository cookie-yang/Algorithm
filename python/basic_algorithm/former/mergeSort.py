def mergeSort(num):
    if len(num)<=1:
        return num
    mid = int(len(num)/2)
    left = mergeSort(num[:mid])
    print("left",num[:mid])
    right = mergeSort(num[mid:])
    print("right", num[mid:])
    return mergesort(left,right)

def mergesort(num1,num2):
    l = 0
    r = 0
    a = []
    while l<len(num1) and r < len(num2):
        print(num1,num2)
        if num1[l]<=num2[r]:
            a.append(num1[l])
            l+=1
        else:
            a.append(num2[r])
            r+=1
    if l < len(num1):
        a+=num1[l:]
    if r < len(num2):
        a+=num2[r:]
    return a




a = [1,25,5,5,6,3,4,3,5,6,8,4,6,2,0]
print(mergeSort(a))
        
