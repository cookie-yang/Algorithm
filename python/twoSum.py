class Solution(object):

    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        newdict = dict()
        num = enumerate(nums)
        for x in num:
            newdict[x[1]] = x[0]
            if (target-x[1]) in newdict and x[0] != newdict.get(target-x[1]):
                return [newdict.get(target - x[1]),x[0]]


nums = [3,3]
target = 6
s = Solution()
# print(s.twoSum(nums,target))
dict2 = {3:2,5:4}
