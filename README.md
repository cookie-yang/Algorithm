# Algorithm
#### This is a note on algorithm on the website [Leetcode](https://leetcode.com) with Java and Python version
#### 1.Two Sum : Given an array of integers, return indices of the two numbers such that they add up to a specific target.
##### Sample:
> Given nums = [2, 7, 11, 15], target = 9,
Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

This problems mainly focused on constructing MAP to reduce time complexity.
In java the method's name is HASHMAP and in python, we can just use dictionary
to achieve it.
The core idea is to construct key-value pair as (elementvalue,index), we can
construct the
