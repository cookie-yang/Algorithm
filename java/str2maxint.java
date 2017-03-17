import java.util.*;
public class str2maxint{
    public boolean compare(String a,String b ){
        String tmp1 = a+b;
        String tmp2 = b+a;
        if(Integer.parseInt(tmp1)>=Integer.parseInt(tmp2))
        return true;
        else
        return false;

    }
    public void swap(String[] str,int a,int b){
        String tmp =new String(str[a]);
        str[a] = str[b];
        str[b] = tmp;
    }
    public void strtomaxint(String[] str,int lo,int up){
        if(lo>up) return;
        else{
            int proviet = lo;
            int left = lo+1;
            int right = up;
            while(left<=right){
                while(left<=right&&compare(str[left],str[proviet])){
                    left++;
                }
                while(left<=right&&compare(str[right],str[proviet])==false){
                    right--;
                }
                if(left>right)break;
                swap(str,left,right);
            }
            swap(str,proviet,right);
            strtomaxint(str,lo,right-1);
            strtomaxint(str,right+1,up);
        }
    }
    public static void main(String[] args){
		String[] str = new String[]{"1","101","100","2","3"};
		List<Integer> strl = new ArrayList<Integer>();
		String str1 = "";
		int cur = 0;
		List<Integer> tmp = new ArrayList<Integer>();
		strtomaxint(str,0,str.length-1);
		StringBuilder sb = new StringBuilder();
		for(int i =0; i<str.length;i++){
			sb.append(str[i]);
		}
		System.out.println(sb);
		
	}
}