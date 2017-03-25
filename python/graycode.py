greycode：the i‘th greycode equals i‘th number xor i’number/2(>>1)
class GrayCode:

    def getGray(self, n):
        # write code here
        i = 0
        list1 = []
        formattext = '{:0' + str(n) + 'b}'
        for x in range(1 << n):
            list1.append(formattext.format(x ^ (x >> 1)))
        return list1
