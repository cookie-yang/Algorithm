public class quick_sort{

    public static void quick_sort(int[] intList,int low,int up){
        if(low>up)return;
        int privot = low;
        int left = low+1;
        int right = up;
        while(left<=right){
            while(left<=right&&intList[left]<intList[privot]){
                left++;
            }
            while(left<=right&&intList[right]>=intList[privot]){
                right--;
            }
            if(left>right)break;
            int tmp = intList[left];
            intList[left] = intList[right];
            intList[right] = tmp;
        }
        int tmp = intList[right];
        intList[right] = intList[privot];
        intList[privot] = tmp;
        quick_sort(intList,low,privot-1);
        quick_sort(intList,privot+1,up);


    }
    public static void main(String[] args){
        int[] intList = new int[]{4,5,6,7,8,1,2,5};
        quick_sort(intList,0,7);
    }
}