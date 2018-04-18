def quickSort(num,lower,upper):
    if lower >= upper:
        return num
    provit = num[lower]
    left = lower+1
    right = upper
    while left<=right:
        while left<=right and num[left]< provit:
            left+=1
        while left<=right and num[right]>=provit:
            right-=1
        if left>right:
            break
        num[left],num[right] = num[right],num[left]
    num[lower],num[right] = num[right],num[lower]
    quickSort(num,lower,right-1)
    quickSort(num,right+1,upper)
    return num


a = [2,3,1,5,6,3,2,8,6,0]
quickSort(a,0,len(a)-1)
print(a)