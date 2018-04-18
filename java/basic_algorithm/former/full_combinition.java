public class full_combinition{
   public static void full_combinition(char[] charlist){
        int num = 1<<charlist.length;
        for(int i = 1;i<num;i++){
        	for(int j = 0;j<charlist.length;j++){
        		if((i&(1<<j))>0)
        			System.out.print(charlist[j]);
        	}
        	System.out.println();
            
        }

    }
    public static void main(String[] args) {
        char[] charList = { 'a', 'b', 'c', 'd' };
        full_combinition(charList);
    }
}