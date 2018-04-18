import java.util.*;
public class longestpalindrome{
    public static int getLongestPalindrom(String s,int i,int j){
        if(i==j) return 1;
        if(i>j) return 0;
        if(s.charAt(i)==s.charAt(j))
          return getLongestPalindrom(s, i+1, j-1)+2;
        else
          return Math.max(getLongestPalindrom(s, i+1, j),getLongestPalindrom(s, i, j-1));
    }
    public static void main(String[] arg){
     Scanner in = new Scanner(System.in);
     String s = in.next();
     System.out.println(getLongestPalindrom(s, 0, s.length()-1));
    }
}