import java.util.*;
public class full_permutation_brute_force{
public static void full_permutation_brute_force(char[] charList,char[] tmp,int[] index,List<String>lstr,int cur){
    	    if(cur == charList.length){
	    	String tmpstr ="";
	        for(int i =0;i<charList.length;i++)
	        tmpstr +=tmp[i];
	        boolean exist = false;
	        for(int i =0;i<lstr.size();i++){
	        	if (lstr.get(i).equals(tmpstr)){
	        		exist = true;
	        	}
	        }
	        if(exist==false){
	        	System.out.println(tmpstr);
	        	lstr.add(tmpstr);
	        	
	        }
	    }
	    else{
	        for(int i = 0;i<charList.length;i++){
	            boolean right = true;
	            
	            for(int j = 0;j<cur;j++){
	                if(index[j]==i)
	                right = false;
	            }
	            if(right){
	                tmp[cur]=charList[i];
	                index[cur] = i;
	                full_permutation_brute_force(charList,tmp,index,lstr,cur+1);
	            }

	        }
	    }

    }
    public static void main(String[] args) {
       char[] charList = {'a','a','b'};
		char[] tmp = new char[3];
		int [] index = new int[3];
		List<String> lstr = new ArrayList<String>();
		full_permutation_brute_force(charList,tmp,index,lstr,0);
    }

}