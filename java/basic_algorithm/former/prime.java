public class prime{
    public static int gcd(int a, int b){
        int d1,d2,d3;
        if(a>=b){
            d1 =a;
            d2 = b;
            d3 = d1%d2;
        }
        else{
            d1 = b;
            d2 = a;
            d3 = d1%d2;
        }
        while(d3!=0){
            d1 = d1/d2;
            d2 = d3;
            d3 = d1%d2;
        }
        return d2;
    }
public static boolean judgePrime(int num){
	for(int i =2;i*i<=num;i++){
		if(num%i==0)return false;
	return num !=1;
	}
}
    public static void main(String[] args) {
     	int m,n;
		Scanner scanner = new Scanner(System.in);
		m = scanner.nextInt();
		n = scanner.nextInt();
		List<String> str1 = new ArrayList();
		List<String> str2 = new ArrayList();
		for(int i=1;i<m;i++){
			for(int j = m;j<=n;j++){
				if (gcd(i,j)==1){
					str1.add("m is"+i+"n is"+j);
				}
				else{
					str2.add("m is"+i+"n is"+j);
				}
					
			}
		  
		}
		 for(int i = 0;i<str1.size();i++){
			   System.out.println(str1.get(i));
		   }
		 System.out.println();
		 for(int i = 0;i<str2.size();i++){
			   System.out.println(str2.get(i));
		   }

    }
}