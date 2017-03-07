import java.util.Map;
import java.util.HashMap;

public class twoSum{
     public static int[] twoSum(int[] nums, int target) {
         Map<Integer,Integer> map = new HashMap<Integer,Integer>();
         for(int i =0;i<nums.length;i++){
             map.put(nums[i],i);
             int remain = target - nums[i];
             if(map.containsKey(remain)&&map.get(remain)!=i){
                 return new int[]{map.get(remain),i};
             }
         }
        throw new IllegalArgumentException("No two sum solution");

}
   public static void main(String []args){
       int []array = {3,3};
       int target = 6;
       int []result = new int[2];
       result = twoSum(array,target);
       System.out.print(result[0]);
       System.out.print(result[1]);
   }
}
